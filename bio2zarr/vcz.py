import abc
import dataclasses
import json
import logging
import os
import pathlib
import shutil

import numcodecs
import numpy as np
import zarr

from bio2zarr import constants, core, provenance, zarr_utils

logger = logging.getLogger(__name__)

ZARR_SCHEMA_FORMAT_VERSION = "0.5"
DEFAULT_ZARR_COMPRESSOR = numcodecs.Blosc(cname="zstd", clevel=7)
DEFAULT_ZARR_COMPRESSOR_GENOTYPES = numcodecs.Blosc(
    cname="zstd", clevel=7, shuffle=numcodecs.Blosc.BITSHUFFLE
)
DEFAULT_ZARR_COMPRESSOR_BOOL = numcodecs.Blosc(
    cname="zstd", clevel=7, shuffle=numcodecs.Blosc.BITSHUFFLE
)

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


class Source(abc.ABC):
    @property
    @abc.abstractmethod
    def path(self):
        pass

    @property
    @abc.abstractmethod
    def num_records(self):
        pass

    @property
    @abc.abstractmethod
    def num_samples(self):
        pass

    @property
    @abc.abstractmethod
    def samples(self):
        pass

    @property
    def contigs(self):
        return None

    @property
    def filters(self):
        return None

    @property
    def root_attrs(self):
        return {}

    @abc.abstractmethod
    def iter_alleles(self, start, stop, num_alleles):
        pass

    @abc.abstractmethod
    def iter_genotypes(self, start, stop, num_alleles):
        pass

    def iter_id(self, start, stop):
        return

    def iter_contig(self, start, stop):
        return

    @abc.abstractmethod
    def iter_field(self, field_name, shape, start, stop):
        pass

    @abc.abstractmethod
    def generate_schema(self, variants_chunk_size, samples_chunk_size, local_alleles):
        pass


@dataclasses.dataclass
class ZarrArraySpec:
    name: str
    dtype: str
    shape: tuple
    chunks: tuple
    dimensions: tuple
    description: str
    compressor: dict = None
    filters: list = None
    source: str = None

    def __post_init__(self):
        if self.name in _fixed_field_descriptions:
            self.description = self.description or _fixed_field_descriptions[self.name]

        # Ensure these are tuples for ease of comparison and consistency
        self.shape = tuple(self.shape)
        self.chunks = tuple(self.chunks)
        self.dimensions = tuple(self.dimensions)
        self.filters = tuple(self.filters) if self.filters is not None else None

    @staticmethod
    def from_field(
        vcf_field,
        *,
        num_variants,
        num_samples,
        variants_chunk_size,
        samples_chunk_size,
        array_name=None,
        compressor=None,
        filters=None,
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
        if (
            vcf_field.summary.max_number > 0 and vcf_field.vcf_number in ("R", "A", "G")
        ) or (vcf_field.summary.max_number > 1 or vcf_field.full_name == "FORMAT/LAA"):
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
        return ZarrArraySpec(
            source=vcf_field.full_name,
            name=array_name,
            dtype=vcf_field.smallest_dtype(),
            shape=shape,
            chunks=chunks,
            dimensions=dimensions,
            description=vcf_field.description,
            compressor=compressor,
            filters=filters,
        )

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
    fields: list
    defaults: dict

    def __init__(
        self,
        format_version: str,
        fields: list,
        variants_chunk_size: int = None,
        samples_chunk_size: int = None,
        defaults: dict = None,
    ):
        self.format_version = format_version
        self.fields = fields
        defaults = defaults.copy() if defaults is not None else {}
        if defaults.get("compressor", None) is None:
            defaults["compressor"] = DEFAULT_ZARR_COMPRESSOR.get_config()
        if defaults.get("filters", None) is None:
            defaults["filters"] = []
        self.defaults = defaults
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
        ret.fields = [ZarrArraySpec(**sd) for sd in d["fields"]]
        return ret

    @staticmethod
    def fromjson(s):
        return VcfZarrSchema.fromdict(json.loads(s))


def sanitise_int_array(value, ndmin, dtype):
    if isinstance(value, tuple):
        value = [
            constants.VCF_INT_MISSING if x is None else x for x in value
        ]  # NEEDS TEST
    value = np.array(value, ndmin=ndmin, copy=True)
    value[value == constants.VCF_INT_MISSING] = -1
    value[value == constants.VCF_INT_FILL] = -2
    # TODO watch out for clipping here!
    return value.astype(dtype)


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
        "call_LAD", "FORMAT/AD", sanitise_int_array, compute_lad_field
    ),
    LocalisableFieldDescriptor(
        "call_LPL", "FORMAT/PL", sanitise_int_array, compute_lpl_field
    ),
]


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
    source_path: str
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


@dataclasses.dataclass
class VcfZarrWriteSummary(core.JsonDataclass):
    num_partitions: int
    num_samples: int
    num_variants: int
    num_chunks: int
    max_encoding_memory: str


class VcfZarrWriter:
    def __init__(self, source_type, path):
        self.source_type = source_type
        self.path = pathlib.Path(path)
        self.wip_path = self.path / "wip"
        self.arrays_path = self.wip_path / "arrays"
        self.partitions_path = self.wip_path / "partitions"
        self.metadata = None
        self.source = None

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
            if field.name == "call_LA" and field.source is None:
                return True
        return False

    #######################
    # init
    #######################

    def init(
        self,
        source,
        *,
        target_num_partitions,
        schema,
        dimension_separator=None,
        max_variant_chunks=None,
    ):
        self.source = source
        if self.path.exists():
            raise ValueError("Zarr path already exists")  # NEEDS TEST
        schema.validate()
        partitions = VcfZarrPartition.generate_partitions(
            self.source.num_records,
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
            source_path=str(self.source.path),
            schema=schema,
            dimension_separator=dimension_separator,
            partitions=partitions,
            # Bare minimum here for provenance - see comments above
            provenance={"source": f"bio2zarr-{provenance.__version__}"},
        )

        self.path.mkdir()
        root = zarr.open(store=self.path, mode="a", **zarr_utils.ZARR_FORMAT_KWARGS)
        root.attrs.update(
            {
                "vcf_zarr_version": "0.2",
                "source": f"bio2zarr-{provenance.__version__}",
            }
        )
        root.attrs.update(self.source.root_attrs)

        # Doing this synchronously - this is fine surely
        self.encode_samples(root)
        if self.source.filters is not None:
            self.encode_filters(root)
        if self.source.contigs is not None:
            self.encode_contigs(root)

        self.wip_path.mkdir()
        self.arrays_path.mkdir()
        self.partitions_path.mkdir()
        root = zarr.open(
            store=self.arrays_path, mode="a", **zarr_utils.ZARR_FORMAT_KWARGS
        )

        total_chunks = 0
        for field in self.schema.fields:
            a = self.init_array(root, self.metadata.schema, field, partitions[-1].stop)
            total_chunks += a.nchunks

        logger.info("Writing WIP metadata")
        with open(self.wip_path / "metadata.json", "w") as f:
            json.dump(self.metadata.asdict(), f, indent=4)

        return VcfZarrWriteSummary(
            num_variants=self.source.num_records,
            num_samples=self.source.num_samples,
            num_partitions=self.num_partitions,
            num_chunks=total_chunks,
            max_encoding_memory=core.display_size(self.get_max_encoding_memory()),
        )

    def encode_samples(self, root):
        samples = self.source.samples
        array = root.array(
            "sample_id",
            data=[sample.id for sample in samples],
            shape=len(samples),
            dtype="str",
            compressor=DEFAULT_ZARR_COMPRESSOR,
            chunks=(self.schema.samples_chunk_size,),
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ["samples"]
        logger.debug("Samples done")

    def encode_contigs(self, root):
        contigs = self.source.contigs
        array = root.array(
            "contig_id",
            data=[contig.id for contig in contigs],
            shape=len(contigs),
            dtype="str",
            compressor=DEFAULT_ZARR_COMPRESSOR,
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ["contigs"]
        if all(contig.length is not None for contig in contigs):
            array = root.array(
                "contig_length",
                data=[contig.length for contig in contigs],
                shape=len(contigs),
                dtype=np.int64,
                compressor=DEFAULT_ZARR_COMPRESSOR,
            )
            array.attrs["_ARRAY_DIMENSIONS"] = ["contigs"]

    def encode_filters(self, root):
        filters = self.source.filters
        array = root.array(
            "filter_id",
            data=[filt.id for filt in filters],
            shape=len(filters),
            dtype="str",
            compressor=DEFAULT_ZARR_COMPRESSOR,
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ["filters"]
        array = root.array(
            "filter_description",
            data=[filt.description for filt in filters],
            shape=len(filters),
            dtype="str",
            compressor=DEFAULT_ZARR_COMPRESSOR,
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ["filters"]

    def init_array(self, root, schema, array_spec, variants_dim_size):
        kwargs = dict(zarr_utils.ZARR_FORMAT_KWARGS)
        filters = (
            array_spec.filters
            if array_spec.filters is not None
            else schema.defaults["filters"]
        )
        filters = [numcodecs.get_codec(filt) for filt in filters]
        compressor = (
            array_spec.compressor
            if array_spec.compressor is not None
            else schema.defaults["compressor"]
        )
        compressor = numcodecs.get_codec(compressor)
        if array_spec.dtype == "O":
            if zarr_utils.zarr_v3():
                filters = [*list(filters), numcodecs.VLenUTF8()]
            else:
                kwargs["object_codec"] = numcodecs.VLenUTF8()

        if not zarr_utils.zarr_v3():
            kwargs["dimension_separator"] = self.metadata.dimension_separator

        shape = list(array_spec.shape)
        # Truncate the variants dimension is max_variant_chunks was specified
        shape[0] = variants_dim_size
        a = root.empty(
            name=array_spec.name,
            shape=shape,
            chunks=array_spec.chunks,
            dtype=array_spec.dtype,
            compressor=compressor,
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
            self.source = self.source_type(self.metadata.source_path)

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

        all_field_names = [field.name for field in self.schema.fields]
        if "variant_id" in all_field_names:
            self.encode_id_partition(partition_index)
        if "variant_filter" in all_field_names:
            self.encode_filters_partition(partition_index)
        if "variant_contig" in all_field_names:
            self.encode_contig_partition(partition_index)
        self.encode_alleles_partition(partition_index)
        for array_spec in self.schema.fields:
            if array_spec.source is not None:
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
        for value in self.source.iter_field(
            array_spec.source,
            ba.buff.shape[1:],
            partition.start,
            partition.stop,
        ):
            j = ba.next_buffer_row()
            ba.buff[j] = value

        self.finalise_partition_array(partition_index, ba)

    def encode_genotypes_partition(self, partition_index):
        partition = self.metadata.partitions[partition_index]
        gt = self.init_partition_array(partition_index, "call_genotype")
        gt_phased = self.init_partition_array(partition_index, "call_genotype_phased")

        for genotype, phased in self.source.iter_genotypes(
            gt.buff.shape[1:], partition.start, partition.stop
        ):
            j = gt.next_buffer_row()
            gt.buff[j] = genotype

            j_phased = gt_phased.next_buffer_row()
            gt_phased.buff[j_phased] = phased

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
            assert field_map[descriptor.array_name].source is None

            buff = self.init_partition_array(partition_index, descriptor.array_name)
            source = self.source.fields[descriptor.vcf_field].iter_values(
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

        for value in self.source.iter_alleles(
            partition.start, partition.stop, alleles.array.shape[1]
        ):
            j = alleles.next_buffer_row()
            alleles.buff[j] = value

        self.finalise_partition_array(partition_index, alleles)

    def encode_id_partition(self, partition_index):
        vid = self.init_partition_array(partition_index, "variant_id")
        vid_mask = self.init_partition_array(partition_index, "variant_id_mask")
        partition = self.metadata.partitions[partition_index]

        for value in self.source.iter_id(partition.start, partition.stop):
            j = vid.next_buffer_row()
            k = vid_mask.next_buffer_row()
            assert j == k
            if value is not None:
                vid.buff[j] = value
                vid_mask.buff[j] = False
            else:
                vid.buff[j] = constants.STR_MISSING
                vid_mask.buff[j] = True

        self.finalise_partition_array(partition_index, vid)
        self.finalise_partition_array(partition_index, vid_mask)

    def encode_filters_partition(self, partition_index):
        var_filter = self.init_partition_array(partition_index, "variant_filter")
        partition = self.metadata.partitions[partition_index]

        for filter_values in self.source.iter_filters(partition.start, partition.stop):
            j = var_filter.next_buffer_row()
            var_filter.buff[j] = filter_values

        self.finalise_partition_array(partition_index, var_filter)

    def encode_contig_partition(self, partition_index):
        contig = self.init_partition_array(partition_index, "variant_contig")
        partition = self.metadata.partitions[partition_index]

        for contig_index in self.source.iter_contig(partition.start, partition.stop):
            j = contig.next_buffer_row()
            contig.buff[j] = contig_index

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

        indexer = VcfZarrIndexer(self.path)
        indexer.create_index()

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
        max_memory = core.parse_max_memory(max_memory)
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


class VcfZarrIndexer:
    """
    Creates an index for efficient region queries in a VCF Zarr dataset.
    """

    def __init__(self, path):
        self.path = pathlib.Path(path)

    def create_index(self):
        """Create an index to support efficient region queries."""
        root = zarr.open_group(store=self.path, mode="r+")

        if (
            "variant_contig" not in root
            or "variant_position" not in root
            or "variant_length" not in root
        ):
            logger.warning("Cannot create index: required arrays not found")
            return

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

        index = np.array(index, dtype=pos.dtype)
        kwargs = {}
        if not zarr_utils.zarr_v3():
            kwargs["dimension_separator"] = "/"
        array = root.array(
            "region_index",
            data=index,
            shape=index.shape,
            chunks=index.shape,
            dtype=index.dtype,
            compressor=numcodecs.Blosc("zstd", clevel=9, shuffle=0),
            fill_value=None,
            **kwargs,
        )
        array.attrs["_ARRAY_DIMENSIONS"] = [
            "region_index_values",
            "region_index_fields",
        ]

        logger.info("Consolidating Zarr metadata")
        zarr.consolidate_metadata(self.path)
