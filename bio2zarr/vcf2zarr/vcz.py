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

from .. import constants, core, provenance
from . import icf

logger = logging.getLogger(__name__)


def inspect(path):
    path = pathlib.Path(path)
    # TODO add support for the Zarr format also
    if (path / "metadata.json").exists():
        obj = icf.IntermediateColumnarFormat(path)
    elif (path / ".zmetadata").exists():
        obj = VcfZarr(path)
    else:
        raise ValueError("Format not recognised")  # NEEDS TEST
    return obj.summary_table()


DEFAULT_ZARR_COMPRESSOR = numcodecs.Blosc(cname="zstd", clevel=7)


@dataclasses.dataclass
class ZarrColumnSpec:
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
        # Ensure these are tuples for ease of comparison and consistency
        self.shape = tuple(self.shape)
        self.chunks = tuple(self.chunks)
        self.dimensions = tuple(self.dimensions)
        self.filters = tuple(self.filters)

    @staticmethod
    def new(**kwargs):
        spec = ZarrColumnSpec(
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
        variable_name=None,
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
        if variable_name is None:
            variable_name = prefix + vcf_field.name
        # TODO make an option to add in the empty extra dimension
        if vcf_field.summary.max_number > 1:
            shape.append(vcf_field.summary.max_number)
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
        return ZarrColumnSpec.new(
            vcf_field=vcf_field.full_name,
            name=variable_name,
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


@dataclasses.dataclass
class VcfZarrSchema(core.JsonDataclass):
    format_version: str
    samples_chunk_size: int
    variants_chunk_size: int
    samples: list
    contigs: list
    filters: list
    fields: list

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
        ret.fields = [ZarrColumnSpec(**sd) for sd in d["fields"]]
        return ret

    @staticmethod
    def fromjson(s):
        return VcfZarrSchema.fromdict(json.loads(s))

    @staticmethod
    def generate(icf, variants_chunk_size=None, samples_chunk_size=None):
        m = icf.num_records
        n = icf.num_samples
        # FIXME
        if samples_chunk_size is None:
            samples_chunk_size = 1000
        if variants_chunk_size is None:
            variants_chunk_size = 10_000
        logger.info(
            f"Generating schema with chunks={variants_chunk_size, samples_chunk_size}"
        )

        def spec_from_field(field, variable_name=None):
            return ZarrColumnSpec.from_field(
                field,
                num_samples=n,
                num_variants=m,
                samples_chunk_size=samples_chunk_size,
                variants_chunk_size=variants_chunk_size,
                variable_name=variable_name,
            )

        def fixed_field_spec(
            name, dtype, vcf_field=None, shape=(m,), dimensions=("variants",)
        ):
            return ZarrColumnSpec.new(
                vcf_field=vcf_field,
                name=name,
                dtype=dtype,
                shape=shape,
                description="",
                dimensions=dimensions,
                chunks=[variants_chunk_size],
            )

        alt_col = icf.fields["ALT"]
        max_alleles = alt_col.vcf_field.summary.max_number + 1

        colspecs = [
            fixed_field_spec(
                name="variant_contig",
                dtype=core.min_int_dtype(0, icf.metadata.num_contigs),
            ),
            fixed_field_spec(
                name="variant_filter",
                dtype="bool",
                shape=(m, icf.metadata.num_filters),
                dimensions=["variants", "filters"],
            ),
            fixed_field_spec(
                name="variant_allele",
                dtype="str",
                shape=(m, max_alleles),
                dimensions=["variants", "alleles"],
            ),
            fixed_field_spec(
                name="variant_id",
                dtype="str",
            ),
            fixed_field_spec(
                name="variant_id_mask",
                dtype="bool",
            ),
        ]
        name_map = {field.full_name: field for field in icf.metadata.fields}

        # Only two of the fixed fields have a direct one-to-one mapping.
        colspecs.extend(
            [
                spec_from_field(name_map["QUAL"], variable_name="variant_quality"),
                spec_from_field(name_map["POS"], variable_name="variant_position"),
            ]
        )
        colspecs.extend([spec_from_field(field) for field in icf.metadata.info_fields])

        gt_field = None
        for field in icf.metadata.format_fields:
            if field.name == "GT":
                gt_field = field
                continue
            colspecs.append(spec_from_field(field))

        if gt_field is not None:
            ploidy = gt_field.summary.max_number - 1
            shape = [m, n]
            chunks = [variants_chunk_size, samples_chunk_size]
            dimensions = ["variants", "samples"]
            colspecs.append(
                ZarrColumnSpec.new(
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
            dimensions += ["ploidy"]
            colspecs.append(
                ZarrColumnSpec.new(
                    vcf_field=None,
                    name="call_genotype",
                    dtype=gt_field.smallest_dtype(),
                    shape=list(shape),
                    chunks=list(chunks),
                    dimensions=list(dimensions),
                    description="",
                )
            )
            colspecs.append(
                ZarrColumnSpec.new(
                    vcf_field=None,
                    name="call_genotype_mask",
                    dtype="bool",
                    shape=list(shape),
                    chunks=list(chunks),
                    dimensions=list(dimensions),
                    description="",
                )
            )

        return VcfZarrSchema(
            format_version=ZARR_SCHEMA_FORMAT_VERSION,
            samples_chunk_size=samples_chunk_size,
            variants_chunk_size=variants_chunk_size,
            fields=colspecs,
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
        store = zarr.DirectoryStore(self.path)
        root = zarr.group(store=store)
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
        store = zarr.DirectoryStore(self.arrays_path)
        root = zarr.group(store=store)

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
            [sample.id for sample in self.schema.samples],
            dtype="str",
            compressor=DEFAULT_ZARR_COMPRESSOR,
            chunks=(self.schema.samples_chunk_size,),
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ["samples"]
        logger.debug("Samples done")

    def encode_contig_id(self, root):
        array = root.array(
            "contig_id",
            [contig.id for contig in self.schema.contigs],
            dtype="str",
            compressor=DEFAULT_ZARR_COMPRESSOR,
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ["contigs"]
        if all(contig.length is not None for contig in self.schema.contigs):
            array = root.array(
                "contig_length",
                [contig.length for contig in self.schema.contigs],
                dtype=np.int64,
                compressor=DEFAULT_ZARR_COMPRESSOR,
            )
            array.attrs["_ARRAY_DIMENSIONS"] = ["contigs"]

    def encode_filter_id(self, root):
        # TODO need a way to store description also
        # https://github.com/sgkit-dev/vcf-zarr-spec/issues/19
        array = root.array(
            "filter_id",
            [filt.id for filt in self.schema.filters],
            dtype="str",
            compressor=DEFAULT_ZARR_COMPRESSOR,
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ["filters"]

    def init_array(self, root, variable, variants_dim_size):
        object_codec = None
        if variable.dtype == "O":
            object_codec = numcodecs.VLenUTF8()
        shape = list(variable.shape)
        # Truncate the variants dimension is max_variant_chunks was specified
        shape[0] = variants_dim_size
        a = root.empty(
            variable.name,
            shape=shape,
            chunks=variable.chunks,
            dtype=variable.dtype,
            compressor=numcodecs.get_codec(variable.compressor),
            filters=[numcodecs.get_codec(filt) for filt in variable.filters],
            object_codec=object_codec,
            dimension_separator=self.metadata.dimension_separator,
        )
        a.attrs.update(
            {
                "description": variable.description,
                # Dimension names are part of the spec in Zarr v3
                "_ARRAY_DIMENSIONS": variable.dimensions,
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
        for col in self.schema.fields:
            if col.vcf_field is not None:
                self.encode_array_partition(col, partition_index)
        if self.has_genotypes():
            self.encode_genotypes_partition(partition_index)

        final_path = self.partition_path(partition_index)
        logger.info(f"Finalising {partition_index} at {final_path}")
        if final_path.exists():
            logger.warning(f"Removing existing partition at {final_path}")
            shutil.rmtree(final_path)
        os.rename(partition_path, final_path)

    def init_partition_array(self, partition_index, name):
        # Create an empty array like the definition
        src = self.arrays_path / name
        # Overwrite any existing WIP files
        wip_path = self.wip_partition_array_path(partition_index, name)
        shutil.copytree(src, wip_path, dirs_exist_ok=True)
        store = zarr.DirectoryStore(self.wip_partition_path(partition_index))
        wip_root = zarr.group(store=store)
        array = wip_root[name]
        logger.debug(f"Opened empty array {array.name} <{array.dtype}> @ {wip_path}")
        return array

    def finalise_partition_array(self, partition_index, name):
        logger.debug(f"Encoded {name} partition {partition_index}")

    def encode_array_partition(self, column, partition_index):
        array = self.init_partition_array(partition_index, column.name)

        partition = self.metadata.partitions[partition_index]
        ba = core.BufferedArray(array, partition.start)
        source_col = self.icf.fields[column.vcf_field]
        sanitiser = source_col.sanitiser_factory(ba.buff.shape)

        for value in source_col.iter_values(partition.start, partition.stop):
            # We write directly into the buffer in the sanitiser function
            # to make it easier to reason about dimension padding
            j = ba.next_buffer_row()
            sanitiser(ba.buff, j, value)
        ba.flush()
        self.finalise_partition_array(partition_index, column.name)

    def encode_genotypes_partition(self, partition_index):
        gt_array = self.init_partition_array(partition_index, "call_genotype")
        gt_mask_array = self.init_partition_array(partition_index, "call_genotype_mask")
        gt_phased_array = self.init_partition_array(
            partition_index, "call_genotype_phased"
        )

        partition = self.metadata.partitions[partition_index]
        gt = core.BufferedArray(gt_array, partition.start)
        gt_mask = core.BufferedArray(gt_mask_array, partition.start)
        gt_phased = core.BufferedArray(gt_phased_array, partition.start)

        source_col = self.icf.fields["FORMAT/GT"]
        for value in source_col.iter_values(partition.start, partition.stop):
            j = gt.next_buffer_row()
            icf.sanitise_value_int_2d(gt.buff, j, value[:, :-1])
            j = gt_phased.next_buffer_row()
            icf.sanitise_value_int_1d(gt_phased.buff, j, value[:, -1])
            # TODO check is this the correct semantics when we are padding
            # with mixed ploidies?
            j = gt_mask.next_buffer_row()
            gt_mask.buff[j] = gt.buff[j] < 0
        gt.flush()
        gt_phased.flush()
        gt_mask.flush()

        self.finalise_partition_array(partition_index, "call_genotype")
        self.finalise_partition_array(partition_index, "call_genotype_mask")
        self.finalise_partition_array(partition_index, "call_genotype_phased")

    def encode_alleles_partition(self, partition_index):
        array_name = "variant_allele"
        alleles_array = self.init_partition_array(partition_index, array_name)
        partition = self.metadata.partitions[partition_index]
        alleles = core.BufferedArray(alleles_array, partition.start)
        ref_col = self.icf.fields["REF"]
        alt_col = self.icf.fields["ALT"]

        for ref, alt in zip(
            ref_col.iter_values(partition.start, partition.stop),
            alt_col.iter_values(partition.start, partition.stop),
        ):
            j = alleles.next_buffer_row()
            alleles.buff[j, :] = constants.STR_FILL
            alleles.buff[j, 0] = ref[0]
            alleles.buff[j, 1 : 1 + len(alt)] = alt
        alleles.flush()

        self.finalise_partition_array(partition_index, array_name)

    def encode_id_partition(self, partition_index):
        vid_array = self.init_partition_array(partition_index, "variant_id")
        vid_mask_array = self.init_partition_array(partition_index, "variant_id_mask")
        partition = self.metadata.partitions[partition_index]
        vid = core.BufferedArray(vid_array, partition.start)
        vid_mask = core.BufferedArray(vid_mask_array, partition.start)
        col = self.icf.fields["ID"]

        for value in col.iter_values(partition.start, partition.stop):
            j = vid.next_buffer_row()
            k = vid_mask.next_buffer_row()
            assert j == k
            if value is not None:
                vid.buff[j] = value[0]
                vid_mask.buff[j] = False
            else:
                vid.buff[j] = constants.STR_MISSING
                vid_mask.buff[j] = True
        vid.flush()
        vid_mask.flush()

        self.finalise_partition_array(partition_index, "variant_id")
        self.finalise_partition_array(partition_index, "variant_id_mask")

    def encode_filters_partition(self, partition_index):
        lookup = {filt.id: index for index, filt in enumerate(self.schema.filters)}
        array_name = "variant_filter"
        array = self.init_partition_array(partition_index, array_name)
        partition = self.metadata.partitions[partition_index]
        var_filter = core.BufferedArray(array, partition.start)

        col = self.icf.fields["FILTERS"]
        for value in col.iter_values(partition.start, partition.stop):
            j = var_filter.next_buffer_row()
            var_filter.buff[j] = False
            for f in value:
                try:
                    var_filter.buff[j, lookup[f]] = True
                except KeyError:
                    raise ValueError(
                        f"Filter '{f}' was not defined in the header."
                    ) from None
        var_filter.flush()

        self.finalise_partition_array(partition_index, array_name)

    def encode_contig_partition(self, partition_index):
        lookup = {contig.id: index for index, contig in enumerate(self.schema.contigs)}
        array_name = "variant_contig"
        array = self.init_partition_array(partition_index, array_name)
        partition = self.metadata.partitions[partition_index]
        contig = core.BufferedArray(array, partition.start)
        col = self.icf.fields["CHROM"]

        for value in col.iter_values(partition.start, partition.stop):
            j = contig.next_buffer_row()
            # Note: because we are using the indexes to define the lookups
            # and we always have an index, it seems that we the contig lookup
            # will always succeed. However, if anyone ever does hit a KeyError
            # here, please do open an issue with a reproducible example!
            contig.buff[j] = lookup[value[0]]
        contig.flush()

        self.finalise_partition_array(partition_index, array_name)

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

    ######################
    # encode_all_partitions
    ######################

    def get_max_encoding_memory(self):
        """
        Return the approximate maximum memory used to encode a variant chunk.
        """
        max_encoding_mem = 0
        for col in self.schema.fields:
            max_encoding_mem = max(max_encoding_mem, col.variant_chunk_nbytes)
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
        for col in self.schema.fields:
            # Open the array definition to get the total size
            total_bytes += zarr.open(self.arrays_path / col.name).nbytes

        progress_config = core.ProgressConfig(
            total=total_bytes,
            title="Encode",
            units="B",
            show=show_progress,
        )
        with core.ParallelWorkManager(num_workers, progress_config) as pwm:
            for partition_index in range(num_partitions):
                pwm.submit(self.encode_partition, partition_index)


def mkschema(if_path, out):
    store = icf.IntermediateColumnarFormat(if_path)
    spec = VcfZarrSchema.generate(store)
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


def encode_init(
    icf_path,
    zarr_path,
    target_num_partitions,
    *,
    schema_path=None,
    variants_chunk_size=None,
    samples_chunk_size=None,
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
    show_progress=False,
    # TODO add arguments to control location of tmpdir
):
    with tempfile.TemporaryDirectory(prefix="vcf2zarr") as tmp:
        if_dir = pathlib.Path(tmp) / "icf"
        icf.explode(
            if_dir,
            vcfs,
            worker_processes=worker_processes,
            show_progress=show_progress,
        )
        encode(
            if_dir,
            out_path,
            variants_chunk_size=variants_chunk_size,
            samples_chunk_size=samples_chunk_size,
            worker_processes=worker_processes,
            show_progress=show_progress,
        )
