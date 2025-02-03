import contextlib
import dataclasses
import json
import logging
import os
import os.path
import pathlib
import shutil
import tempfile

import humanfriendly
import numcodecs
import numpy as np
import zarr

from bio2zarr.zarr_utils import ZARR_FORMAT_KWARGS, zarr_v3

from .. import constants, core, provenance
from . import icf

logger = logging.getLogger(__name__)


def inspect(path):
    path = pathlib.Path(path)
    if not path.exists():
        raise ValueError(f"Path not found: {path}")
    if (path / "metadata.json").exists():
        obj = icf.IntermediateColumnarFormat(path)
    # NOTE: this is too strict, we should support more general Zarrs, see #276
    elif (path / ".zmetadata").exists():
        obj = VcfZarr(path)
    else:
        raise ValueError(f"{path} not in ICF or VCF Zarr format")
    return obj.summary_table()


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


ZARR_SCHEMA_FORMAT_VERSION = "0.4"


def convert_local_allele_field_types(fields):
    """
    Update the specified list of fields to include the LAA field, and to convert
    any supported localisable fields to the L* counterpart.

    Note that we currently support only two ALT alleles per sample, and so the
    dimensions of these fields are fixed by that requirement. Later versions may
    use summry data storted in the ICF to make different choices, if information
    about subsequent alleles (not in the actual genotype calls) should also be
    stored.
    """
    fields_by_name = {field.name: field for field in fields}
    gt = fields_by_name["call_genotype"]
    if gt.shape[-1] != 2:
        raise ValueError("Local alleles only supported on diploid data")

    # TODO check if LA is already in here

    shape = gt.shape[:-1]
    chunks = gt.chunks[:-1]

    la = ZarrArraySpec.new(
        vcf_field=None,
        name="call_LA",
        dtype="i1",
        shape=gt.shape,
        chunks=gt.chunks,
        dimensions=gt.dimensions,  # FIXME
        description=(
            "0-based indices into REF+ALT, indicating which alleles"
            " are relevant (local) for the current sample"
        ),
    )
    ad = fields_by_name.get("call_AD", None)
    if ad is not None:
        # TODO check if call_LAD is in the list already
        ad.name = "call_LAD"
        ad.vcf_field = None
        ad.shape = (*shape, 2)
        ad.chunks = (*chunks, 2)
        ad.description += " (local-alleles)"
        # TODO fix dimensions

    pl = fields_by_name.get("call_PL", None)
    if pl is not None:
        # TODO check if call_LPL is in the list already
        pl.name = "call_LPL"
        pl.vcf_field = None
        pl.shape = (*shape, 3)
        pl.chunks = (*chunks, 3)
        pl.description += " (local-alleles)"
        # TODO fix dimensions
    return [*fields, la]


@dataclasses.dataclass
class VcfZarrSchema(core.JsonDataclass):
    format_version: str
    samples_chunk_size: int
    variants_chunk_size: int
    samples: list
    contigs: list
    filters: list
    fields: list

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
        ret.samples = [icf.Sample(**sd) for sd in d["samples"]]
        ret.contigs = [icf.Contig(**sd) for sd in d["contigs"]]
        ret.filters = [icf.Filter(**sd) for sd in d["filters"]]
        ret.fields = [ZarrArraySpec(**sd) for sd in d["fields"]]
        return ret

    @staticmethod
    def fromjson(s):
        return VcfZarrSchema.fromdict(json.loads(s))

    @staticmethod
    def generate(
        icf, variants_chunk_size=None, samples_chunk_size=None, local_alleles=None
    ):
        m = icf.num_records
        n = icf.num_samples
        if samples_chunk_size is None:
            samples_chunk_size = 10_000
        if variants_chunk_size is None:
            variants_chunk_size = 1000
        if local_alleles is None:
            local_alleles = False
        logger.info(
            f"Generating schema with chunks={variants_chunk_size, samples_chunk_size}"
        )

        def spec_from_field(field, array_name=None):
            return ZarrArraySpec.from_field(
                field,
                num_samples=n,
                num_variants=m,
                samples_chunk_size=samples_chunk_size,
                variants_chunk_size=variants_chunk_size,
                array_name=array_name,
            )

        def fixed_field_spec(
            name,
            dtype,
            vcf_field=None,
            shape=(m,),
            dimensions=("variants",),
            chunks=None,
        ):
            return ZarrArraySpec.new(
                vcf_field=vcf_field,
                name=name,
                dtype=dtype,
                shape=shape,
                description="",
                dimensions=dimensions,
                chunks=chunks or [variants_chunk_size],
            )

        alt_field = icf.fields["ALT"]
        max_alleles = alt_field.vcf_field.summary.max_number + 1

        array_specs = [
            fixed_field_spec(
                name="variant_contig",
                dtype=core.min_int_dtype(0, icf.metadata.num_contigs),
            ),
            fixed_field_spec(
                name="variant_filter",
                dtype="bool",
                shape=(m, icf.metadata.num_filters),
                dimensions=["variants", "filters"],
                chunks=(variants_chunk_size, icf.metadata.num_filters),
            ),
            fixed_field_spec(
                name="variant_allele",
                dtype="O",
                shape=(m, max_alleles),
                dimensions=["variants", "alleles"],
                chunks=(variants_chunk_size, max_alleles),
            ),
            fixed_field_spec(
                name="variant_id",
                dtype="O",
            ),
            fixed_field_spec(
                name="variant_id_mask",
                dtype="bool",
            ),
        ]
        name_map = {field.full_name: field for field in icf.metadata.fields}

        # Only three of the fixed fields have a direct one-to-one mapping.
        array_specs.extend(
            [
                spec_from_field(name_map["QUAL"], array_name="variant_quality"),
                spec_from_field(name_map["POS"], array_name="variant_position"),
                spec_from_field(name_map["rlen"], array_name="variant_length"),
            ]
        )
        array_specs.extend(
            [spec_from_field(field) for field in icf.metadata.info_fields]
        )

        gt_field = None
        for field in icf.metadata.format_fields:
            if field.name == "GT":
                gt_field = field
                continue
            array_specs.append(spec_from_field(field))

        if gt_field is not None and n > 0:
            ploidy = max(gt_field.summary.max_number - 1, 1)
            shape = [m, n]
            chunks = [variants_chunk_size, samples_chunk_size]
            dimensions = ["variants", "samples"]
            array_specs.append(
                ZarrArraySpec.new(
                    vcf_field=None,
                    name="call_genotype_phased",
                    dtype="bool",
                    shape=list(shape),
                    chunks=list(chunks),
                    dimensions=list(dimensions),
                    description="",
                )
            )
            shape += [ploidy]
            chunks += [ploidy]
            dimensions += ["ploidy"]
            array_specs.append(
                ZarrArraySpec.new(
                    vcf_field=None,
                    name="call_genotype",
                    dtype=gt_field.smallest_dtype(),
                    shape=list(shape),
                    chunks=list(chunks),
                    dimensions=list(dimensions),
                    description="",
                )
            )
            array_specs.append(
                ZarrArraySpec.new(
                    vcf_field=None,
                    name="call_genotype_mask",
                    dtype="bool",
                    shape=list(shape),
                    chunks=list(chunks),
                    dimensions=list(dimensions),
                    description="",
                )
            )

        if local_alleles:
            array_specs = convert_local_allele_field_types(array_specs)

        return VcfZarrSchema(
            format_version=ZARR_SCHEMA_FORMAT_VERSION,
            samples_chunk_size=samples_chunk_size,
            variants_chunk_size=variants_chunk_size,
            fields=array_specs,
            samples=icf.metadata.samples,
            contigs=icf.metadata.contigs,
            filters=icf.metadata.filters,
        )


class VcfZarr:
    def __init__(self, path):
        if not (path / ".zmetadata").exists():
            raise ValueError("Not in VcfZarr format")  # NEEDS TEST
        self.path = path
        self.root = zarr.open(path, mode="r")

    def summary_table(self):
        data = []
        arrays = [(core.du(self.path / a.basename), a) for _, a in self.root.arrays()]
        arrays.sort(key=lambda x: x[0])
        for stored, array in reversed(arrays):
            d = {
                "name": array.name,
                "dtype": str(array.dtype),
                "stored": core.display_size(stored),
                "size": core.display_size(array.nbytes),
                "ratio": core.display_number(array.nbytes / stored),
                "nchunks": str(array.nchunks),
                "chunk_size": core.display_size(array.nbytes / array.nchunks),
                "avg_chunk_stored": core.display_size(int(stored / array.nchunks)),
                "shape": str(array.shape),
                "chunk_shape": str(array.chunks),
                "compressor": str(array.compressor),
                "filters": str(array.filters),
            }
            data.append(d)
        return data


def parse_max_memory(max_memory):
    if max_memory is None:
        # Effectively unbounded
        return 2**63
    if isinstance(max_memory, str):
        max_memory = humanfriendly.parse_size(max_memory)
    logger.info(f"Set memory budget to {core.display_size(max_memory)}")
    return max_memory


@dataclasses.dataclass
class VcfZarrPartition:
    start: int
    stop: int

    @staticmethod
    def generate_partitions(num_records, chunk_size, num_partitions, max_chunks=None):
        num_chunks = int(np.ceil(num_records / chunk_size))
        if max_chunks is not None:
            num_chunks = min(num_chunks, max_chunks)
        partitions = []
        splits = np.array_split(np.arange(num_chunks), min(num_partitions, num_chunks))
        for chunk_slice in splits:
            start_chunk = int(chunk_slice[0])
            stop_chunk = int(chunk_slice[-1]) + 1
            start_index = start_chunk * chunk_size
            stop_index = min(stop_chunk * chunk_size, num_records)
            partitions.append(VcfZarrPartition(start_index, stop_index))
        return partitions


VZW_METADATA_FORMAT_VERSION = "0.1"


@dataclasses.dataclass
class VcfZarrWriterMetadata(core.JsonDataclass):
    format_version: str
    icf_path: str
    schema: VcfZarrSchema
    dimension_separator: str
    partitions: list
    provenance: dict

    @staticmethod
    def fromdict(d):
        if d["format_version"] != VZW_METADATA_FORMAT_VERSION:
            raise ValueError(
                "VcfZarrWriter format version mismatch: "
                f"{d['format_version']} != {VZW_METADATA_FORMAT_VERSION}"
            )
        ret = VcfZarrWriterMetadata(**d)
        ret.schema = VcfZarrSchema.fromdict(ret.schema)
        ret.partitions = [VcfZarrPartition(**p) for p in ret.partitions]
        return ret


def compute_la_field(genotypes):
    """
    Computes the value of the LA field for each sample given the genotypes
    for a variant. The LA field lists the unique alleles observed for
    each sample, including the REF.
    """
    v = 2**31 - 1
    if np.any(genotypes >= v):
        raise ValueError("Extreme allele value not supported")
    G = genotypes.astype(np.int32)
    if len(G) > 0:
        # Anything < 0 gets mapped to -2 (pad) in the output, which comes last.
        # So, to get this sorting correctly, we remap to the largest value for
        # sorting, then map back. We promote the genotypes up to 32 bit for convenience
        # here, assuming that we'll never have a allele of 2**31 - 1.
        assert np.all(G != v)
        G[G < 0] = v
        G.sort(axis=1)
        G[G[:, 0] == G[:, 1], 1] = -2
        # Equal values result in padding also
        G[G == v] = -2
    return G.astype(genotypes.dtype)


def compute_lad_field(ad, la):
    assert ad.shape[0] == la.shape[0]
    assert la.shape[1] == 2
    lad = np.full((ad.shape[0], 2), -2, dtype=ad.dtype)
    homs = np.where((la[:, 0] != -2) & (la[:, 1] == -2))
    lad[homs, 0] = ad[homs, la[homs, 0]]
    hets = np.where(la[:, 1] != -2)
    lad[hets, 0] = ad[hets, la[hets, 0]]
    lad[hets, 1] = ad[hets, la[hets, 1]]
    return lad


def pl_index(a, b):
    """
    Returns the PL index for alleles a and b.
    """
    return b * (b + 1) // 2 + a


def compute_lpl_field(pl, la):
    lpl = np.full((pl.shape[0], 3), -2, dtype=pl.dtype)

    homs = np.where((la[:, 0] != -2) & (la[:, 1] == -2))
    a = la[homs, 0]
    lpl[homs, 0] = pl[homs, pl_index(a, a)]

    hets = np.where(la[:, 1] != -2)[0]
    a = la[hets, 0]
    b = la[hets, 1]
    lpl[hets, 0] = pl[hets, pl_index(a, a)]
    lpl[hets, 1] = pl[hets, pl_index(a, b)]
    lpl[hets, 2] = pl[hets, pl_index(b, b)]

    return lpl


@dataclasses.dataclass
class LocalisableFieldDescriptor:
    array_name: str
    vcf_field: str
    sanitise: callable
    convert: callable


localisable_fields = [
    LocalisableFieldDescriptor(
        "call_LAD", "FORMAT/AD", icf.sanitise_int_array, compute_lad_field
    ),
    LocalisableFieldDescriptor(
        "call_LPL", "FORMAT/PL", icf.sanitise_int_array, compute_lpl_field
    ),
]


@dataclasses.dataclass
class VcfZarrWriteSummary(core.JsonDataclass):
    num_partitions: int
    num_samples: int
    num_variants: int
    num_chunks: int
    max_encoding_memory: str


class VcfZarrWriter:
    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.wip_path = self.path / "wip"
        self.arrays_path = self.wip_path / "arrays"
        self.partitions_path = self.wip_path / "partitions"
        self.metadata = None
        self.icf = None

    @property
    def schema(self):
        return self.metadata.schema

    @property
    def num_partitions(self):
        return len(self.metadata.partitions)

    def has_genotypes(self):
        for field in self.schema.fields:
            if field.name == "call_genotype":
                return True
        return False

    def has_local_alleles(self):
        for field in self.schema.fields:
            if field.name == "call_LA" and field.vcf_field is None:
                return True
        return False

    #######################
    # init
    #######################

    def init(
        self,
        icf,
        *,
        target_num_partitions,
        schema,
        dimension_separator=None,
        max_variant_chunks=None,
    ):
        self.icf = icf
        if self.path.exists():
            raise ValueError("Zarr path already exists")  # NEEDS TEST
        schema.validate()
        partitions = VcfZarrPartition.generate_partitions(
            self.icf.num_records,
            schema.variants_chunk_size,
            target_num_partitions,
            max_chunks=max_variant_chunks,
        )
        # Default to using nested directories following the Zarr v3 default.
        # This seems to require version 2.17+ to work properly
        dimension_separator = (
            "/" if dimension_separator is None else dimension_separator
        )
        self.metadata = VcfZarrWriterMetadata(
            format_version=VZW_METADATA_FORMAT_VERSION,
            icf_path=str(self.icf.path),
            schema=schema,
            dimension_separator=dimension_separator,
            partitions=partitions,
            # Bare minimum here for provenance - see comments above
            provenance={"source": f"bio2zarr-{provenance.__version__}"},
        )

        self.path.mkdir()
        root = zarr.open(store=self.path, mode="a", **ZARR_FORMAT_KWARGS)
        root.attrs.update(
            {
                "vcf_zarr_version": "0.2",
                "vcf_header": self.icf.vcf_header,
                "source": f"bio2zarr-{provenance.__version__}",
            }
        )
        # Doing this syncronously - this is fine surely
        self.encode_samples(root)
        self.encode_filter_id(root)
        self.encode_contig_id(root)

        self.wip_path.mkdir()
        self.arrays_path.mkdir()
        self.partitions_path.mkdir()
        root = zarr.open(store=self.arrays_path, mode="a", **ZARR_FORMAT_KWARGS)

        total_chunks = 0
        for field in self.schema.fields:
            a = self.init_array(root, field, partitions[-1].stop)
            total_chunks += a.nchunks

        logger.info("Writing WIP metadata")
        with open(self.wip_path / "metadata.json", "w") as f:
            json.dump(self.metadata.asdict(), f, indent=4)

        return VcfZarrWriteSummary(
            num_variants=self.icf.num_records,
            num_samples=self.icf.num_samples,
            num_partitions=self.num_partitions,
            num_chunks=total_chunks,
            max_encoding_memory=core.display_size(self.get_max_encoding_memory()),
        )

    def encode_samples(self, root):
        if self.schema.samples != self.icf.metadata.samples:
            raise ValueError("Subsetting or reordering samples not supported currently")
        array = root.array(
            "sample_id",
            data=[sample.id for sample in self.schema.samples],
            shape=len(self.schema.samples),
            dtype="str",
            compressor=DEFAULT_ZARR_COMPRESSOR,
            chunks=(self.schema.samples_chunk_size,),
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ["samples"]
        logger.debug("Samples done")

    def encode_contig_id(self, root):
        array = root.array(
            "contig_id",
            data=[contig.id for contig in self.schema.contigs],
            shape=len(self.schema.contigs),
            dtype="str",
            compressor=DEFAULT_ZARR_COMPRESSOR,
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ["contigs"]
        if all(contig.length is not None for contig in self.schema.contigs):
            array = root.array(
                "contig_length",
                data=[contig.length for contig in self.schema.contigs],
                shape=len(self.schema.contigs),
                dtype=np.int64,
                compressor=DEFAULT_ZARR_COMPRESSOR,
            )
            array.attrs["_ARRAY_DIMENSIONS"] = ["contigs"]

    def encode_filter_id(self, root):
        # TODO need a way to store description also
        # https://github.com/sgkit-dev/vcf-zarr-spec/issues/19
        array = root.array(
            "filter_id",
            data=[filt.id for filt in self.schema.filters],
            shape=len(self.schema.filters),
            dtype="str",
            compressor=DEFAULT_ZARR_COMPRESSOR,
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ["filters"]

    def init_array(self, root, array_spec, variants_dim_size):
        kwargs = dict(ZARR_FORMAT_KWARGS)
        filters = [numcodecs.get_codec(filt) for filt in array_spec.filters]
        if array_spec.dtype == "O":
            if zarr_v3():
                filters = [*list(filters), numcodecs.VLenUTF8()]
            else:
                kwargs["object_codec"] = numcodecs.VLenUTF8()

        if not zarr_v3():
            kwargs["dimension_separator"] = self.metadata.dimension_separator

        shape = list(array_spec.shape)
        # Truncate the variants dimension is max_variant_chunks was specified
        shape[0] = variants_dim_size
        a = root.empty(
            name=array_spec.name,
            shape=shape,
            chunks=array_spec.chunks,
            dtype=array_spec.dtype,
            compressor=numcodecs.get_codec(array_spec.compressor),
            filters=filters,
            **kwargs,
        )
        a.attrs.update(
            {
                "description": array_spec.description,
                # Dimension names are part of the spec in Zarr v3
                "_ARRAY_DIMENSIONS": array_spec.dimensions,
            }
        )
        logger.debug(f"Initialised {a}")
        return a

    #######################
    # encode_partition
    #######################

    def load_metadata(self):
        if self.metadata is None:
            with open(self.wip_path / "metadata.json") as f:
                self.metadata = VcfZarrWriterMetadata.fromdict(json.load(f))
            self.icf = icf.IntermediateColumnarFormat(self.metadata.icf_path)

    def partition_path(self, partition_index):
        return self.partitions_path / f"p{partition_index}"

    def wip_partition_path(self, partition_index):
        return self.partitions_path / f"wip_p{partition_index}"

    def wip_partition_array_path(self, partition_index, name):
        return self.wip_partition_path(partition_index) / name

    def partition_array_path(self, partition_index, name):
        return self.partition_path(partition_index) / name

    def encode_partition(self, partition_index):
        self.load_metadata()
        if partition_index < 0 or partition_index >= self.num_partitions:
            raise ValueError("Partition index not in the valid range")
        partition_path = self.wip_partition_path(partition_index)
        partition_path.mkdir(exist_ok=True)
        logger.info(f"Encoding partition {partition_index} to {partition_path}")

        self.encode_id_partition(partition_index)
        self.encode_filters_partition(partition_index)
        self.encode_contig_partition(partition_index)
        self.encode_alleles_partition(partition_index)
        for array_spec in self.schema.fields:
            if array_spec.vcf_field is not None:
                self.encode_array_partition(array_spec, partition_index)
        if self.has_genotypes():
            self.encode_genotypes_partition(partition_index)
            self.encode_genotype_mask_partition(partition_index)
        if self.has_local_alleles():
            self.encode_local_alleles_partition(partition_index)
            self.encode_local_allele_fields_partition(partition_index)

        final_path = self.partition_path(partition_index)
        logger.info(f"Finalising {partition_index} at {final_path}")
        if final_path.exists():
            logger.warning(f"Removing existing partition at {final_path}")
            shutil.rmtree(final_path)
        os.rename(partition_path, final_path)

    def init_partition_array(self, partition_index, name):
        field_map = self.schema.field_map()
        array_spec = field_map[name]
        # Create an empty array like the definition
        src = self.arrays_path / array_spec.name
        # Overwrite any existing WIP files
        wip_path = self.wip_partition_array_path(partition_index, array_spec.name)
        shutil.copytree(src, wip_path, dirs_exist_ok=True)
        array = zarr.open_array(store=wip_path, mode="a")
        partition = self.metadata.partitions[partition_index]
        ba = core.BufferedArray(array, partition.start, name)
        logger.info(
            f"Start partition {partition_index} array {name} <{array.dtype}> "
            f"{array.shape} @ {wip_path}"
        )
        return ba

    def finalise_partition_array(self, partition_index, buffered_array):
        buffered_array.flush()
        logger.info(
            f"Completed partition {partition_index} array {buffered_array.name} "
            f"max_memory={core.display_size(buffered_array.max_buff_size)}"
        )

    def encode_array_partition(self, array_spec, partition_index):
        partition = self.metadata.partitions[partition_index]
        ba = self.init_partition_array(partition_index, array_spec.name)
        source_field = self.icf.fields[array_spec.vcf_field]
        sanitiser = source_field.sanitiser_factory(ba.buff.shape)

        for value in source_field.iter_values(partition.start, partition.stop):
            # We write directly into the buffer in the sanitiser function
            # to make it easier to reason about dimension padding
            j = ba.next_buffer_row()
            sanitiser(ba.buff, j, value)
        self.finalise_partition_array(partition_index, ba)

    def encode_genotypes_partition(self, partition_index):
        partition = self.metadata.partitions[partition_index]
        gt = self.init_partition_array(partition_index, "call_genotype")
        gt_phased = self.init_partition_array(partition_index, "call_genotype_phased")

        source_field = self.icf.fields["FORMAT/GT"]
        for value in source_field.iter_values(partition.start, partition.stop):
            j = gt.next_buffer_row()
            icf.sanitise_value_int_2d(
                gt.buff, j, value[:, :-1] if value is not None else None
            )
            j = gt_phased.next_buffer_row()
            icf.sanitise_value_int_1d(
                gt_phased.buff, j, value[:, -1] if value is not None else None
            )

        self.finalise_partition_array(partition_index, gt)
        self.finalise_partition_array(partition_index, gt_phased)

    def encode_genotype_mask_partition(self, partition_index):
        partition = self.metadata.partitions[partition_index]
        gt_mask = self.init_partition_array(partition_index, "call_genotype_mask")
        # Read back in the genotypes so we can compute the mask
        gt_array = zarr.open_array(
            store=self.wip_partition_array_path(partition_index, "call_genotype"),
            mode="r",
        )
        for genotypes in core.first_dim_slice_iter(
            gt_array, partition.start, partition.stop
        ):
            # TODO check is this the correct semantics when we are padding
            # with mixed ploidies?
            j = gt_mask.next_buffer_row()
            gt_mask.buff[j] = genotypes < 0
        self.finalise_partition_array(partition_index, gt_mask)

    def encode_local_alleles_partition(self, partition_index):
        partition = self.metadata.partitions[partition_index]
        call_LA = self.init_partition_array(partition_index, "call_LA")

        gt_array = zarr.open_array(
            store=self.wip_partition_array_path(partition_index, "call_genotype"),
            mode="r",
        )
        for genotypes in core.first_dim_slice_iter(
            gt_array, partition.start, partition.stop
        ):
            la = compute_la_field(genotypes)
            j = call_LA.next_buffer_row()
            call_LA.buff[j] = la
        self.finalise_partition_array(partition_index, call_LA)

    def encode_local_allele_fields_partition(self, partition_index):
        partition = self.metadata.partitions[partition_index]
        la_array = zarr.open_array(
            store=self.wip_partition_array_path(partition_index, "call_LA"),
            mode="r",
        )
        # We got through the localisable fields one-by-one so that we don't need to
        # keep several large arrays in memory at once for each partition.
        field_map = self.schema.field_map()
        for descriptor in localisable_fields:
            if descriptor.array_name not in field_map:
                continue
            assert field_map[descriptor.array_name].vcf_field is None

            buff = self.init_partition_array(partition_index, descriptor.array_name)
            source = self.icf.fields[descriptor.vcf_field].iter_values(
                partition.start, partition.stop
            )
            for la in core.first_dim_slice_iter(
                la_array, partition.start, partition.stop
            ):
                raw_value = next(source)
                value = descriptor.sanitise(raw_value, 2, raw_value.dtype)
                j = buff.next_buffer_row()
                buff.buff[j] = descriptor.convert(value, la)
            self.finalise_partition_array(partition_index, buff)

    def encode_alleles_partition(self, partition_index):
        alleles = self.init_partition_array(partition_index, "variant_allele")
        partition = self.metadata.partitions[partition_index]
        ref_field = self.icf.fields["REF"]
        alt_field = self.icf.fields["ALT"]

        for ref, alt in zip(
            ref_field.iter_values(partition.start, partition.stop),
            alt_field.iter_values(partition.start, partition.stop),
        ):
            j = alleles.next_buffer_row()
            alleles.buff[j, :] = constants.STR_FILL
            alleles.buff[j, 0] = ref[0]
            alleles.buff[j, 1 : 1 + len(alt)] = alt
        self.finalise_partition_array(partition_index, alleles)

    def encode_id_partition(self, partition_index):
        vid = self.init_partition_array(partition_index, "variant_id")
        vid_mask = self.init_partition_array(partition_index, "variant_id_mask")
        partition = self.metadata.partitions[partition_index]
        field = self.icf.fields["ID"]

        for value in field.iter_values(partition.start, partition.stop):
            j = vid.next_buffer_row()
            k = vid_mask.next_buffer_row()
            assert j == k
            if value is not None:
                vid.buff[j] = value[0]
                vid_mask.buff[j] = False
            else:
                vid.buff[j] = constants.STR_MISSING
                vid_mask.buff[j] = True

        self.finalise_partition_array(partition_index, vid)
        self.finalise_partition_array(partition_index, vid_mask)

    def encode_filters_partition(self, partition_index):
        lookup = {filt.id: index for index, filt in enumerate(self.schema.filters)}
        var_filter = self.init_partition_array(partition_index, "variant_filter")
        partition = self.metadata.partitions[partition_index]

        field = self.icf.fields["FILTERS"]
        for value in field.iter_values(partition.start, partition.stop):
            j = var_filter.next_buffer_row()
            var_filter.buff[j] = False
            for f in value:
                try:
                    var_filter.buff[j, lookup[f]] = True
                except KeyError:
                    raise ValueError(
                        f"Filter '{f}' was not defined in the header."
                    ) from None

        self.finalise_partition_array(partition_index, var_filter)

    def encode_contig_partition(self, partition_index):
        lookup = {contig.id: index for index, contig in enumerate(self.schema.contigs)}
        contig = self.init_partition_array(partition_index, "variant_contig")
        partition = self.metadata.partitions[partition_index]
        field = self.icf.fields["CHROM"]

        for value in field.iter_values(partition.start, partition.stop):
            j = contig.next_buffer_row()
            # Note: because we are using the indexes to define the lookups
            # and we always have an index, it seems that we the contig lookup
            # will always succeed. However, if anyone ever does hit a KeyError
            # here, please do open an issue with a reproducible example!
            contig.buff[j] = lookup[value[0]]

        self.finalise_partition_array(partition_index, contig)

    #######################
    # finalise
    #######################

    def finalise_array(self, name):
        logger.info(f"Finalising {name}")
        final_path = self.path / name
        if final_path.exists():
            # NEEDS TEST
            raise ValueError(f"Array {name} already exists")
        for partition in range(self.num_partitions):
            # Move all the files in partition dir to dest dir
            src = self.partition_array_path(partition, name)
            if not src.exists():
                # Needs test
                raise ValueError(f"Partition {partition} of {name} does not exist")
            dest = self.arrays_path / name
            # This is Zarr v2 specific. Chunks in v3 with start with "c" prefix.
            chunk_files = [
                path for path in src.iterdir() if not path.name.startswith(".")
            ]
            # TODO check for a count of then number of files. If we require a
            # dimension_separator of "/" then we could make stronger assertions
            # here, as we'd always have num_variant_chunks
            logger.debug(
                f"Moving {len(chunk_files)} chunks for {name} partition {partition}"
            )
            for chunk_file in chunk_files:
                os.rename(chunk_file, dest / chunk_file.name)
        # Finally, once all the chunks have moved into the arrays dir,
        # we move it out of wip
        os.rename(self.arrays_path / name, self.path / name)
        core.update_progress(1)

    def finalise(self, show_progress=False):
        self.load_metadata()

        logger.info(f"Scanning {self.num_partitions} partitions")
        missing = []
        # TODO may need a progress bar here
        for partition_id in range(self.num_partitions):
            if not self.partition_path(partition_id).exists():
                missing.append(partition_id)
        if len(missing) > 0:
            raise FileNotFoundError(f"Partitions not encoded: {missing}")

        progress_config = core.ProgressConfig(
            total=len(self.schema.fields),
            title="Finalise",
            units="array",
            show=show_progress,
        )
        # NOTE: it's not clear that adding more workers will make this quicker,
        # as it's just going to be causing contention on the file system.
        # Something to check empirically in some deployments.
        # FIXME we're just using worker_processes=0 here to hook into the
        # SynchronousExecutor which is intended for testing purposes so
        # that we get test coverage. Should fix this either by allowing
        # for multiple workers, or making a standard wrapper for tqdm
        # that allows us to have a consistent look and feel.
        with core.ParallelWorkManager(0, progress_config) as pwm:
            for field in self.schema.fields:
                pwm.submit(self.finalise_array, field.name)
        logger.debug(f"Removing {self.wip_path}")
        shutil.rmtree(self.wip_path)
        logger.info("Consolidating Zarr metadata")
        zarr.consolidate_metadata(self.path)

    #######################
    # index
    #######################

    def create_index(self):
        """Create an index to support efficient region queries."""

        root = zarr.open_group(store=self.path, mode="r+")

        contig = root["variant_contig"]
        pos = root["variant_position"]
        length = root["variant_length"]

        assert contig.cdata_shape == pos.cdata_shape

        index = []

        logger.info("Creating region index")
        for v_chunk in range(pos.cdata_shape[0]):
            c = contig.blocks[v_chunk]
            p = pos.blocks[v_chunk]
            e = p + length.blocks[v_chunk] - 1

            # create a row for each contig in the chunk
            d = np.diff(c, append=-1)
            c_start_idx = 0
            for c_end_idx in np.nonzero(d)[0]:
                assert c[c_start_idx] == c[c_end_idx]
                index.append(
                    (
                        v_chunk,  # chunk index
                        c[c_start_idx],  # contig ID
                        p[c_start_idx],  # start
                        p[c_end_idx],  # end
                        np.max(e[c_start_idx : c_end_idx + 1]),  # max end
                        c_end_idx - c_start_idx + 1,  # num records
                    )
                )
                c_start_idx = c_end_idx + 1

        index = np.array(index, dtype=np.int32)
        kwargs = {}
        if not zarr_v3():
            kwargs["dimension_separator"] = self.metadata.dimension_separator
        array = root.array(
            "region_index",
            data=index,
            shape=index.shape,
            dtype=index.dtype,
            compressor=numcodecs.Blosc("zstd", clevel=9, shuffle=0),
            **kwargs,
        )
        array.attrs["_ARRAY_DIMENSIONS"] = [
            "region_index_values",
            "region_index_fields",
        ]

        logger.info("Consolidating Zarr metadata")
        zarr.consolidate_metadata(self.path)

    ######################
    # encode_all_partitions
    ######################

    def get_max_encoding_memory(self):
        """
        Return the approximate maximum memory used to encode a variant chunk.
        """
        max_encoding_mem = 0
        for array_spec in self.schema.fields:
            max_encoding_mem = max(max_encoding_mem, array_spec.variant_chunk_nbytes)
        gt_mem = 0
        if self.has_genotypes:
            gt_mem = sum(
                field.variant_chunk_nbytes
                for field in self.schema.fields
                if field.name.startswith("call_genotype")
            )
        return max(max_encoding_mem, gt_mem)

    def encode_all_partitions(
        self, *, worker_processes=1, show_progress=False, max_memory=None
    ):
        max_memory = parse_max_memory(max_memory)
        self.load_metadata()
        num_partitions = self.num_partitions
        per_worker_memory = self.get_max_encoding_memory()
        logger.info(
            f"Encoding Zarr over {num_partitions} partitions with "
            f"{worker_processes} workers and {core.display_size(per_worker_memory)} "
            "per worker"
        )
        # Each partition requires per_worker_memory bytes, so to prevent more that
        # max_memory being used, we clamp the number of workers
        max_num_workers = max_memory // per_worker_memory
        if max_num_workers < worker_processes:
            logger.warning(
                f"Limiting number of workers to {max_num_workers} to "
                "keep within specified memory budget of "
                f"{core.display_size(max_memory)}"
            )
        if max_num_workers <= 0:
            raise ValueError(
                f"Insufficient memory to encode a partition:"
                f"{core.display_size(per_worker_memory)} > "
                f"{core.display_size(max_memory)}"
            )
        num_workers = min(max_num_workers, worker_processes)

        total_bytes = 0
        for array_spec in self.schema.fields:
            # Open the array definition to get the total size
            total_bytes += zarr.open(self.arrays_path / array_spec.name).nbytes

        progress_config = core.ProgressConfig(
            total=total_bytes,
            title="Encode",
            units="B",
            show=show_progress,
        )
        with core.ParallelWorkManager(num_workers, progress_config) as pwm:
            for partition_index in range(num_partitions):
                pwm.submit(self.encode_partition, partition_index)


def mkschema(
    if_path,
    out,
    *,
    variants_chunk_size=None,
    samples_chunk_size=None,
    local_alleles=None,
):
    store = icf.IntermediateColumnarFormat(if_path)
    spec = VcfZarrSchema.generate(
        store,
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
        local_alleles=local_alleles,
    )
    out.write(spec.asjson())


def encode(
    if_path,
    zarr_path,
    schema_path=None,
    variants_chunk_size=None,
    samples_chunk_size=None,
    max_variant_chunks=None,
    dimension_separator=None,
    max_memory=None,
    local_alleles=None,
    worker_processes=1,
    show_progress=False,
):
    # Rough heuristic to split work up enough to keep utilisation high
    target_num_partitions = max(1, worker_processes * 4)
    encode_init(
        if_path,
        zarr_path,
        target_num_partitions,
        schema_path=schema_path,
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
        local_alleles=local_alleles,
        max_variant_chunks=max_variant_chunks,
        dimension_separator=dimension_separator,
    )
    vzw = VcfZarrWriter(zarr_path)
    vzw.encode_all_partitions(
        worker_processes=worker_processes,
        show_progress=show_progress,
        max_memory=max_memory,
    )
    vzw.finalise(show_progress)
    vzw.create_index()


def encode_init(
    icf_path,
    zarr_path,
    target_num_partitions,
    *,
    schema_path=None,
    variants_chunk_size=None,
    samples_chunk_size=None,
    local_alleles=None,
    max_variant_chunks=None,
    dimension_separator=None,
    max_memory=None,
    worker_processes=1,
    show_progress=False,
):
    icf_store = icf.IntermediateColumnarFormat(icf_path)
    if schema_path is None:
        schema = VcfZarrSchema.generate(
            icf_store,
            variants_chunk_size=variants_chunk_size,
            samples_chunk_size=samples_chunk_size,
            local_alleles=local_alleles,
        )
    else:
        logger.info(f"Reading schema from {schema_path}")
        if variants_chunk_size is not None or samples_chunk_size is not None:
            raise ValueError(
                "Cannot specify schema along with chunk sizes"
            )  # NEEDS TEST
        with open(schema_path) as f:
            schema = VcfZarrSchema.fromjson(f.read())
    zarr_path = pathlib.Path(zarr_path)
    vzw = VcfZarrWriter(zarr_path)
    return vzw.init(
        icf_store,
        target_num_partitions=target_num_partitions,
        schema=schema,
        dimension_separator=dimension_separator,
        max_variant_chunks=max_variant_chunks,
    )


def encode_partition(zarr_path, partition):
    writer = VcfZarrWriter(zarr_path)
    writer.encode_partition(partition)


def encode_finalise(zarr_path, show_progress=False):
    writer = VcfZarrWriter(zarr_path)
    writer.finalise(show_progress=show_progress)


def convert(
    vcfs,
    out_path,
    *,
    variants_chunk_size=None,
    samples_chunk_size=None,
    worker_processes=1,
    local_alleles=None,
    show_progress=False,
    icf_path=None,
):
    if icf_path is None:
        cm = temp_icf_path(prefix="vcf2zarr")
    else:
        cm = contextlib.nullcontext(icf_path)

    with cm as icf_path:
        icf.explode(
            icf_path,
            vcfs,
            worker_processes=worker_processes,
            show_progress=show_progress,
        )
        encode(
            icf_path,
            out_path,
            variants_chunk_size=variants_chunk_size,
            samples_chunk_size=samples_chunk_size,
            worker_processes=worker_processes,
            show_progress=show_progress,
            local_alleles=local_alleles,
        )


@contextlib.contextmanager
def temp_icf_path(prefix=None):
    with tempfile.TemporaryDirectory(prefix=prefix) as tmp:
        yield pathlib.Path(tmp) / "icf"
