import dataclasses
import json
import logging

import numcodecs
import numpy as np

from bio2zarr import core

logger = logging.getLogger(__name__)

ZARR_SCHEMA_FORMAT_VERSION = "0.4"

DEFAULT_ZARR_COMPRESSOR = numcodecs.Blosc(cname="zstd", clevel=7)

_fixed_field_descriptions = {
    "variant_contig": "An identifier from the reference genome or an angle-bracketed ID"
    " string pointing to a contig in the assembly file",
    "variant_position": "The reference position",
    "variant_length": "The length of the variant measured in bases",
    "variant_id": "List of unique identifiers where applicable",
    "variant_allele": "List of the reference and alternate alleles",
    "variant_quality": "Phred-scaled quality score",
    "variant_filter": "Filter status of the variant",
}


@dataclasses.dataclass
class ZarrArraySpec:
    name: str
    dtype: str
    shape: tuple
    chunks: tuple
    dimensions: tuple
    description: str
    vcf_field: str
    compressor: dict
    filters: list

    def __post_init__(self):
        if self.name in _fixed_field_descriptions:
            self.description = self.description or _fixed_field_descriptions[self.name]

        # Ensure these are tuples for ease of comparison and consistency
        self.shape = tuple(self.shape)
        self.chunks = tuple(self.chunks)
        self.dimensions = tuple(self.dimensions)
        self.filters = tuple(self.filters)

    @staticmethod
    def new(**kwargs):
        spec = ZarrArraySpec(
            **kwargs, compressor=DEFAULT_ZARR_COMPRESSOR.get_config(), filters=[]
        )
        spec._choose_compressor_settings()
        return spec

    @staticmethod
    def from_field(
        vcf_field,
        *,
        num_variants,
        num_samples,
        variants_chunk_size,
        samples_chunk_size,
        array_name=None,
    ):
        shape = [num_variants]
        prefix = "variant_"
        dimensions = ["variants"]
        chunks = [variants_chunk_size]
        if vcf_field.category == "FORMAT":
            prefix = "call_"
            shape.append(num_samples)
            chunks.append(samples_chunk_size)
            dimensions.append("samples")
        if array_name is None:
            array_name = prefix + vcf_field.name
        # TODO make an option to add in the empty extra dimension
        if vcf_field.summary.max_number > 1 or vcf_field.full_name == "FORMAT/LAA":
            shape.append(vcf_field.summary.max_number)
            chunks.append(vcf_field.summary.max_number)
            # TODO we should really be checking this to see if the named dimensions
            # are actually correct.
            if vcf_field.vcf_number == "R":
                dimensions.append("alleles")
            elif vcf_field.vcf_number == "A":
                dimensions.append("alt_alleles")
            elif vcf_field.vcf_number == "G":
                dimensions.append("genotypes")
            else:
                dimensions.append(f"{vcf_field.category}_{vcf_field.name}_dim")
        return ZarrArraySpec.new(
            vcf_field=vcf_field.full_name,
            name=array_name,
            dtype=vcf_field.smallest_dtype(),
            shape=shape,
            chunks=chunks,
            dimensions=dimensions,
            description=vcf_field.description,
        )

    def _choose_compressor_settings(self):
        """
        Choose compressor and filter settings based on the size and
        type of the array, plus some hueristics from observed properties
        of VCFs.

        See https://github.com/pystatgen/bio2zarr/discussions/74
        """
        # Default is to not shuffle, because autoshuffle isn't recognised
        # by many Zarr implementations, and shuffling can lead to worse
        # performance in some cases anyway. Turning on shuffle should be a
        # deliberate choice.
        shuffle = numcodecs.Blosc.NOSHUFFLE
        if self.name == "call_genotype" and self.dtype == "i1":
            # call_genotype gets BITSHUFFLE by default as it gets
            # significantly better compression (at a cost of slower
            # decoding)
            shuffle = numcodecs.Blosc.BITSHUFFLE
        elif self.dtype == "bool":
            shuffle = numcodecs.Blosc.BITSHUFFLE

        self.compressor["shuffle"] = shuffle

    @property
    def chunk_nbytes(self):
        """
        Returns the nbytes for a single chunk in this array.
        """
        items = 1
        dim = 0
        for chunk_size in self.chunks:
            size = min(chunk_size, self.shape[dim])
            items *= size
            dim += 1
        # Include sizes for extra dimensions.
        for size in self.shape[dim:]:
            items *= size
        dt = np.dtype(self.dtype)
        return items * dt.itemsize

    @property
    def variant_chunk_nbytes(self):
        """
        Returns the nbytes for a single variant chunk of this array.
        """
        chunk_items = self.chunks[0]
        for size in self.shape[1:]:
            chunk_items *= size
        dt = np.dtype(self.dtype)
        if dt.kind == "O" and "samples" in self.dimensions:
            logger.warning(
                f"Field {self.name} is a string; max memory usage may "
                "be a significant underestimate"
            )
        return chunk_items * dt.itemsize


@dataclasses.dataclass
class Contig:
    id: str
    length: int = None


@dataclasses.dataclass
class Sample:
    id: str


@dataclasses.dataclass
class Filter:
    id: str
    description: str = ""


@dataclasses.dataclass
class VcfZarrSchema(core.JsonDataclass):
    format_version: str
    samples_chunk_size: int
    variants_chunk_size: int
    samples: list
    contigs: list
    filters: list
    fields: list

    def __init__(
        self,
        format_version: str,
        samples: list,
        contigs: list,
        filters: list,
        fields: list,
        variants_chunk_size: int = None,
        samples_chunk_size: int = None,
    ):
        self.format_version = format_version
        self.samples = samples
        self.contigs = contigs
        self.filters = filters
        self.fields = fields
        if variants_chunk_size is None:
            variants_chunk_size = 1000
        self.variants_chunk_size = variants_chunk_size
        if samples_chunk_size is None:
            samples_chunk_size = 10_000
        self.samples_chunk_size = samples_chunk_size

    def validate(self):
        """
        Checks that the schema is well-formed and within required limits.
        """
        for field in self.fields:
            # This is the Blosc max buffer size
            if field.chunk_nbytes > 2147483647:
                # TODO add some links to documentation here advising how to
                # deal with PL values.
                raise ValueError(
                    f"Field {field.name} chunks are too large "
                    f"({field.chunk_nbytes} > 2**31 - 1 bytes). "
                    "Either generate a schema and drop this field (if you don't "
                    "need it) or reduce the variant or sample chunk sizes."
                )
            # TODO other checks? There must be lots of ways people could mess
            # up the schema leading to cryptic errors.

    def field_map(self):
        return {field.name: field for field in self.fields}

    @staticmethod
    def fromdict(d):
        if d["format_version"] != ZARR_SCHEMA_FORMAT_VERSION:
            raise ValueError(
                "Zarr schema format version mismatch: "
                f"{d['format_version']} != {ZARR_SCHEMA_FORMAT_VERSION}"
            )
        ret = VcfZarrSchema(**d)
        ret.samples = [Sample(**sd) for sd in d["samples"]]
        ret.contigs = [Contig(**sd) for sd in d["contigs"]]
        ret.filters = [Filter(**sd) for sd in d["filters"]]
        ret.fields = [ZarrArraySpec(**sd) for sd in d["fields"]]
        return ret

    @staticmethod
    def fromjson(s):
        return VcfZarrSchema.fromdict(json.loads(s))
