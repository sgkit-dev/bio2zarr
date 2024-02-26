import concurrent.futures as cf
import collections
import dataclasses
import functools
import logging
import os
import pathlib
import pickle
import sys
import shutil
import json
import math
import tempfile
import contextlib
from typing import Any

import humanfriendly
import cyvcf2
import numcodecs
import numpy as np
import numpy.testing as nt
import tqdm
import zarr

from . import core
from . import provenance

logger = logging.getLogger(__name__)

INT_MISSING = -1
INT_FILL = -2
STR_MISSING = "."
STR_FILL = ""

FLOAT32_MISSING, FLOAT32_FILL = np.array([0x7F800001, 0x7F800002], dtype=np.int32).view(
    np.float32
)
FLOAT32_MISSING_AS_INT32, FLOAT32_FILL_AS_INT32 = np.array(
    [0x7F800001, 0x7F800002], dtype=np.int32
)


@dataclasses.dataclass
class VcfFieldSummary:
    num_chunks: int = 0
    compressed_size: int = 0
    uncompressed_size: int = 0
    max_number: int = 0  # Corresponds to VCF Number field, depends on context
    # Only defined for numeric fields
    max_value: Any = -math.inf
    min_value: Any = math.inf

    def update(self, other):
        self.num_chunks += other.num_chunks
        self.compressed_size += other.compressed_size
        self.uncompressed_size = other.uncompressed_size
        self.max_number = max(self.max_number, other.max_number)
        self.min_value = min(self.min_value, other.min_value)
        self.max_value = max(self.max_value, other.max_value)


@dataclasses.dataclass
class VcfField:
    category: str
    name: str
    vcf_number: str
    vcf_type: str
    description: str
    summary: VcfFieldSummary

    @staticmethod
    def from_header(definition):
        category = definition["HeaderType"]
        name = definition["ID"]
        vcf_number = definition["Number"]
        vcf_type = definition["Type"]
        return VcfField(
            category=category,
            name=name,
            vcf_number=vcf_number,
            vcf_type=vcf_type,
            description=definition["Description"].strip('"'),
            summary=VcfFieldSummary(),
        )

    @staticmethod
    def fromdict(d):
        f = VcfField(**d)
        f.summary = VcfFieldSummary(**d["summary"])
        return f

    @property
    def full_name(self):
        if self.category == "fixed":
            return self.name
        return f"{self.category}/{self.name}"

    # TODO add method here to choose a good set compressor and
    # filters default here for this field.

    def smallest_dtype(self):
        """
        Returns the smallest dtype suitable for this field based
        on type, and values.
        """
        s = self.summary
        if self.vcf_type == "Float":
            ret = "f4"
        elif self.vcf_type == "Integer":
            dtype = "i4"
            for a_dtype in ["i1", "i2"]:
                info = np.iinfo(a_dtype)
                if info.min <= s.min_value and s.max_value <= info.max:
                    dtype = a_dtype
                    break
            ret = dtype
        elif self.vcf_type == "Flag":
            ret = "bool"
        elif self.vcf_type == "Character":
            ret = "U1"
        else:
            assert self.vcf_type == "String"
            ret = "O"
        return ret


@dataclasses.dataclass
class VcfPartition:
    vcf_path: str
    num_records: int
    first_position: int


@dataclasses.dataclass
class VcfMetadata:
    samples: list
    contig_names: list
    filters: list
    fields: list
    contig_lengths: list = None
    partitions: list = None

    @staticmethod
    def fromdict(d):
        fields = [VcfField.fromdict(fd) for fd in d["fields"]]
        partitions = [VcfPartition(**pd) for pd in d["partitions"]]
        d = d.copy()
        d["fields"] = fields
        d["partitions"] = partitions
        return VcfMetadata(**d)

    def asdict(self):
        return dataclasses.asdict(self)


def fixed_vcf_field_definitions():
    def make_field_def(name, vcf_type, vcf_number):
        return VcfField(
            category="fixed",
            name=name,
            vcf_type=vcf_type,
            vcf_number=vcf_number,
            description="",
            summary=VcfFieldSummary(),
        )

    fields = [
        make_field_def("CHROM", "String", "1"),
        make_field_def("POS", "Integer", "1"),
        make_field_def("QUAL", "Float", "1"),
        make_field_def("ID", "String", "."),
        make_field_def("FILTERS", "String", "."),
        make_field_def("REF", "String", "1"),
        make_field_def("ALT", "String", "."),
    ]
    return fields


def scan_vcfs(paths, show_progress):
    partitions = []
    vcf_metadata = None
    header = None
    logger.info(f"Scanning {len(paths)} VCFs")
    for path in tqdm.tqdm(paths, desc="Scan ", disable=not show_progress):
        vcf = cyvcf2.VCF(path)
        logger.debug(f"Scanning {path}")

        filters = [
            h["ID"]
            for h in vcf.header_iter()
            if h["HeaderType"] == "FILTER" and isinstance(h["ID"], str)
        ]
        # Ensure PASS is the first filter if present
        if "PASS" in filters:
            filters.remove("PASS")
            filters.insert(0, "PASS")

        fields = fixed_vcf_field_definitions()
        for h in vcf.header_iter():
            if h["HeaderType"] in ["INFO", "FORMAT"]:
                field = VcfField.from_header(h)
                if field.name == "GT":
                    field.vcf_type = "Integer"
                    field.vcf_number = "."
                fields.append(field)

        metadata = VcfMetadata(
            samples=vcf.samples,
            contig_names=vcf.seqnames,
            filters=filters,
            fields=fields,
        )
        try:
            metadata.contig_lengths = vcf.seqlens
        except AttributeError:
            pass

        if vcf_metadata is None:
            vcf_metadata = metadata
            # We just take the first header, assuming the others
            # are compatible.
            header = vcf.raw_header
        else:
            if metadata != vcf_metadata:
                raise ValueError("Incompatible VCF chunks")
        record = next(vcf)

        partitions.append(
            # Requires cyvcf2>=0.30.27
            VcfPartition(
                vcf_path=str(path),
                num_records=vcf.num_records,
                first_position=(record.CHROM, record.POS),
            )
        )
    partitions.sort(key=lambda x: x.first_position)
    vcf_metadata.partitions = partitions
    return vcf_metadata, header


def sanitise_value_bool(buff, j, value):
    x = True
    if value is None:
        x = False
    buff[j] = x


def sanitise_value_float_scalar(buff, j, value):
    x = value
    if value is None:
        x = FLOAT32_MISSING
    buff[j] = x


def sanitise_value_int_scalar(buff, j, value):
    x = value
    if value is None:
        # print("MISSING", INT_MISSING, INT_FILL)
        x = [INT_MISSING]
    else:
        x = sanitise_int_array([value], ndmin=1, dtype=np.int32)
    buff[j] = x[0]


def sanitise_value_string_scalar(buff, j, value):
    if value is None:
        buff[j] = "."
    else:
        buff[j] = value[0]


def sanitise_value_string_1d(buff, j, value):
    if value is None:
        buff[j] = "."
    else:
        # value = np.array(value, ndmin=1, dtype=buff.dtype, copy=False)
        # FIXME failure isn't coming from here, it seems to be from an
        # incorrectly detected dimension in the zarr array
        # The dimesions look all wrong, and the dtype should be Object
        # not str
        value = drop_empty_second_dim(value)
        buff[j] = ""
        buff[j, : value.shape[0]] = value


def sanitise_value_string_2d(buff, j, value):
    if value is None:
        buff[j] = "."
    else:
        # print(buff.shape, value.dtype, value)
        # assert value.ndim == 2
        buff[j] = ""
        if value.ndim == 2:
            buff[j, :, : value.shape[1]] = value
        else:
            # TODO check if this is still necessary
            for k, val in enumerate(value):
                buff[j, k, : len(val)] = val


def drop_empty_second_dim(value):
    assert len(value.shape) == 1 or value.shape[1] == 1
    if len(value.shape) == 2 and value.shape[1] == 1:
        value = value[..., 0]
    return value


def sanitise_value_float_1d(buff, j, value):
    if value is None:
        buff[j] = FLOAT32_MISSING
    else:
        value = np.array(value, ndmin=1, dtype=buff.dtype, copy=False)
        # numpy will map None values to Nan, but we need a
        # specific NaN
        value[np.isnan(value)] = FLOAT32_MISSING
        value = drop_empty_second_dim(value)
        buff[j] = FLOAT32_FILL
        buff[j, : value.shape[0]] = value


def sanitise_value_float_2d(buff, j, value):
    if value is None:
        buff[j] = FLOAT32_MISSING
    else:
        # print("value = ", value)
        value = np.array(value, ndmin=2, dtype=buff.dtype, copy=False)
        buff[j] = FLOAT32_FILL
        buff[j, :, : value.shape[1]] = value


def sanitise_int_array(value, ndmin, dtype):
    if isinstance(value, tuple):
        value = [VCF_INT_MISSING if x is None else x for x in value]
    value = np.array(value, ndmin=ndmin, copy=False)
    value[value == VCF_INT_MISSING] = -1
    value[value == VCF_INT_FILL] = -2
    # TODO watch out for clipping here!
    return value.astype(dtype)


def sanitise_value_int_1d(buff, j, value):
    if value is None:
        buff[j] = -1
    else:
        value = sanitise_int_array(value, 1, buff.dtype)
        value = drop_empty_second_dim(value)
        buff[j] = -2
        buff[j, : value.shape[0]] = value


def sanitise_value_int_2d(buff, j, value):
    if value is None:
        buff[j] = -1
    else:
        value = sanitise_int_array(value, 2, buff.dtype)
        buff[j] = -2
        buff[j, :, : value.shape[1]] = value


MIN_INT_VALUE = np.iinfo(np.int32).min + 2
VCF_INT_MISSING = np.iinfo(np.int32).min
VCF_INT_FILL = np.iinfo(np.int32).min + 1

missing_value_map = {
    "Integer": -1,
    "Float": FLOAT32_MISSING,
    "String": ".",
    "Character": ".",
    "Flag": False,
}


class VcfValueTransformer:
    """
    Transform VCF values into the stored intermediate format used
    in the PickleChunkedVcf, and update field summaries.
    """

    def __init__(self, field, num_samples):
        self.field = field
        self.num_samples = num_samples
        self.dimension = 1
        if field.category == "FORMAT":
            self.dimension = 2
        self.missing = missing_value_map[field.vcf_type]

    @staticmethod
    def factory(field, num_samples):
        if field.vcf_type in ("Integer", "Flag"):
            return IntegerValueTransformer(field, num_samples)
        if field.vcf_type == "Float":
            return FloatValueTransformer(field, num_samples)
        if field.name in ["REF", "FILTERS", "ALT", "ID", "CHROM"]:
            return SplitStringValueTransformer(field, num_samples)
        return StringValueTransformer(field, num_samples)

    def transform(self, vcf_value):
        if isinstance(vcf_value, tuple):
            vcf_value = [self.missing if v is None else v for v in vcf_value]
        value = np.array(vcf_value, ndmin=self.dimension, copy=False)
        return value

    def transform_and_update_bounds(self, vcf_value):
        if vcf_value is None:
            return None
        value = self.transform(vcf_value)
        self.update_bounds(value)
        # print(self.field.full_name, "T", vcf_value, "->", value)
        return value


MIN_INT_VALUE = np.iinfo(np.int32).min + 2
VCF_INT_MISSING = np.iinfo(np.int32).min
VCF_INT_FILL = np.iinfo(np.int32).min + 1


class IntegerValueTransformer(VcfValueTransformer):
    def update_bounds(self, value):
        summary = self.field.summary
        # Mask out missing and fill values
        # print(value)
        a = value[value >= MIN_INT_VALUE]
        if a.size > 0:
            summary.max_value = int(max(summary.max_value, np.max(a)))
            summary.min_value = int(min(summary.min_value, np.min(a)))
        number = value.shape[-1]
        summary.max_number = max(summary.max_number, number)


class FloatValueTransformer(VcfValueTransformer):
    def update_bounds(self, value):
        summary = self.field.summary
        summary.max_value = float(max(summary.max_value, np.max(value)))
        summary.min_value = float(min(summary.min_value, np.min(value)))
        number = value.shape[-1]
        summary.max_number = max(summary.max_number, number)


class StringValueTransformer(VcfValueTransformer):
    def update_bounds(self, value):
        summary = self.field.summary
        number = value.shape[-1]
        # TODO would be nice to report string lengths, but not
        # really necessary.
        summary.max_number = max(summary.max_number, number)

    def transform(self, vcf_value):
        # print("transform", vcf_value)
        if self.dimension == 1:
            value = np.array(list(vcf_value.split(",")))
        else:
            # TODO can we make this faster??
            value = np.array([v.split(",") for v in vcf_value], dtype="O")
            # print("HERE", vcf_value, value)
            # for v in vcf_value:
            #     print("\t", type(v), len(v), v.split(","))
        # print("S: ", self.dimension, ":", value.shape, value)
        return value


class SplitStringValueTransformer(StringValueTransformer):
    def transform(self, vcf_value):
        if vcf_value is None:
            return self.missing_value
        assert self.dimension == 1
        return np.array(vcf_value, ndmin=1, dtype="str")


class PickleChunkedVcfField:
    def __init__(self, vcf_field, base_path):
        self.vcf_field = vcf_field
        if vcf_field.category == "fixed":
            self.path = base_path / vcf_field.name
        else:
            self.path = base_path / vcf_field.category / vcf_field.name

        # TODO Check if other compressors would give reasonable compression
        # with significantly faster times
        self.compressor = numcodecs.Blosc(cname="zstd", clevel=7)
        # TODO have a clearer way of defining this state between
        # read and write mode.
        self.num_partitions = None
        self.num_records = None
        self.partition_num_chunks = {}

    def __repr__(self):
        return f"PickleChunkedVcfField(path={self.path})"

    def num_chunks(self, partition_index):
        if partition_index not in self.partition_num_chunks:
            partition_path = self.path / f"p{partition_index}"
            n = len(list(partition_path.iterdir()))
            self.partition_num_chunks[partition_index] = n
        return self.partition_num_chunks[partition_index]

    def chunk_path(self, partition_index, chunk_index):
        return self.path / f"p{partition_index}" / f"c{chunk_index}"

    def write_chunk(self, partition_index, chunk_index, data):
        path = self.chunk_path(partition_index, chunk_index)
        logger.debug(f"Start write: {path}")
        pkl = pickle.dumps(data)
        # NOTE assuming that reusing the same compressor instance
        # from multiple threads is OK!
        compressed = self.compressor.encode(pkl)
        with open(path, "wb") as f:
            f.write(compressed)

        # Update the summary
        self.vcf_field.summary.num_chunks += 1
        self.vcf_field.summary.compressed_size += len(compressed)
        self.vcf_field.summary.uncompressed_size += len(pkl)
        logger.debug(f"Finish write: {path}")

    def read_chunk(self, partition_index, chunk_index):
        path = self.chunk_path(partition_index, chunk_index)
        with open(path, "rb") as f:
            pkl = self.compressor.decode(f.read())
        return pickle.loads(pkl), len(pkl)

    def iter_values_bytes(self):
        num_records = 0
        bytes_read = 0
        for partition_index in range(self.num_partitions):
            for chunk_index in range(self.num_chunks(partition_index)):
                chunk, chunk_bytes = self.read_chunk(partition_index, chunk_index)
                bytes_read += chunk_bytes
                for record in chunk:
                    yield record, bytes_read
                    num_records += 1
        if num_records != self.num_records:
            raise ValueError(
                f"Corruption detected: incorrect number of records in {str(self.path)}."
            )

    # Note: this involves some computation so should arguably be a method,
    # but making a property for consistency with xarray etc
    @property
    def values(self):
        return [record for record, _ in self.iter_values_bytes()]

    def sanitiser_factory(self, shape):
        """
        Return a function that sanitised values from this column
        and writes into a buffer of the specified shape.
        """
        assert len(shape) <= 3
        if self.vcf_field.vcf_type == "Flag":
            assert len(shape) == 1
            return sanitise_value_bool
        elif self.vcf_field.vcf_type == "Float":
            if len(shape) == 1:
                return sanitise_value_float_scalar
            elif len(shape) == 2:
                return sanitise_value_float_1d
            else:
                return sanitise_value_float_2d
        elif self.vcf_field.vcf_type == "Integer":
            if len(shape) == 1:
                return sanitise_value_int_scalar
            elif len(shape) == 2:
                return sanitise_value_int_1d
            else:
                return sanitise_value_int_2d
        else:
            assert self.vcf_field.vcf_type in ("String", "Character")
            if len(shape) == 1:
                return sanitise_value_string_scalar
            elif len(shape) == 2:
                return sanitise_value_string_1d
            else:
                return sanitise_value_string_2d


@dataclasses.dataclass
class FieldBuffer:
    field: PickleChunkedVcfField
    transformer: VcfValueTransformer
    buff: list = dataclasses.field(default_factory=list)
    buffered_bytes: int = 0
    chunk_index: int = 0

    def append(self, val):
        self.buff.append(val)
        val_bytes = sys.getsizeof(val)
        self.buffered_bytes += val_bytes

    def reset(self):
        self.buff = []
        self.buffered_bytes = 0
        self.chunk_index += 1


class ThreadedColumnWriter(contextlib.AbstractContextManager):
    def __init__(
        self,
        vcf_metadata,
        out_path,
        partition_index,
        *,
        encoder_threads=0,
        chunk_size=1,
    ):
        self.encoder_threads = encoder_threads
        self.partition_index = partition_index
        # chunk_size is in megabytes
        self.max_buffered_bytes = chunk_size * 2**20
        assert self.max_buffered_bytes > 0

        if encoder_threads <= 0:
            # NOTE: this is only for testing, not for production use!
            self.executor = core.SynchronousExecutor()
        else:
            self.executor = cf.ProcessPoolExecutor(max_workers=encoder_threads)

        self.buffers = {}
        num_samples = len(vcf_metadata.samples)
        for vcf_field in vcf_metadata.fields:
            field = PickleChunkedVcfField(vcf_field, out_path)
            transformer = VcfValueTransformer.factory(vcf_field, num_samples)
            self.buffers[vcf_field.full_name] = FieldBuffer(field, transformer)
        self.futures = set()

    @property
    def field_summaries(self):
        return {
            name: buff.field.vcf_field.summary for name, buff in self.buffers.items()
        }

    def append(self, name, value):
        buff = self.buffers[name]
        # print("Append", name, value)
        value = buff.transformer.transform_and_update_bounds(value)
        assert value is None or isinstance(value, np.ndarray)
        buff.append(value)
        val_bytes = sys.getsizeof(value)
        buff.buffered_bytes += val_bytes
        if buff.buffered_bytes >= self.max_buffered_bytes:
            self._flush_buffer(name, buff)

    def _service_futures(self):
        max_waiting = 2 * self.encoder_threads
        while len(self.futures) > max_waiting:
            futures_done, _ = cf.wait(self.futures, return_when=cf.FIRST_COMPLETED)
            for future in futures_done:
                exception = future.exception()
                if exception is not None:
                    raise exception
                self.futures.remove(future)

    def _flush_buffer(self, name, buff):
        self._service_futures()
        logger.debug(f"Schedule write {name}:{self.partition_index}.{buff.chunk_index}")
        future = self.executor.submit(
            buff.field.write_chunk,
            self.partition_index,
            buff.chunk_index,
            buff.buff,
        )
        self.futures.add(future)
        buff.reset()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Normal exit condition
            for name, buff in self.buffers.items():
                self._flush_buffer(name, buff)
            core.wait_on_futures(self.futures)
        else:
            core.cancel_futures(self.futures)
        self.executor.shutdown()
        return False


class PickleChunkedVcf(collections.abc.Mapping):
    def __init__(self, path, metadata, vcf_header):
        self.path = path
        self.metadata = metadata
        self.vcf_header = vcf_header

        self.columns = {}
        for field in self.metadata.fields:
            self.columns[field.full_name] = PickleChunkedVcfField(field, path)

        for col in self.columns.values():
            col.num_partitions = self.num_partitions
            col.num_records = self.num_records

    def __getitem__(self, key):
        return self.columns[key]

    def __iter__(self):
        return iter(self.columns)

    def __len__(self):
        return len(self.columns)

    def summary_table(self):
        def display_number(x):
            ret = "n/a"
            if math.isfinite(x):
                ret = f"{x: 0.2g}"
            return ret

        def display_size(n):
            return humanfriendly.format_size(n)

        data = []
        for name, col in self.columns.items():
            summary = col.vcf_field.summary
            d = {
                "name": name,
                "type": col.vcf_field.vcf_type,
                "chunks": summary.num_chunks,
                "size": display_size(summary.uncompressed_size),
                "compressed": display_size(summary.compressed_size),
                "max_n": summary.max_number,
                "min_val": display_number(summary.min_value),
                "max_val": display_number(summary.max_value),
            }

            data.append(d)
        return data

    @functools.cached_property
    def total_uncompressed_bytes(self):
        total = 0
        for col in self.columns.values():
            summary = col.vcf_field.summary
            total += summary.uncompressed_size
        return total

    @functools.cached_property
    def num_records(self):
        return sum(partition.num_records for partition in self.metadata.partitions)

    @property
    def num_partitions(self):
        return len(self.metadata.partitions)

    @property
    def num_samples(self):
        return len(self.metadata.samples)

    @property
    def num_columns(self):
        return len(self.columns)

    def mkdirs(self):
        self.path.mkdir()
        for col in self.columns.values():
            col.path.mkdir(parents=True)
            for j in range(self.num_partitions):
                part_path = col.path / f"p{j}"
                part_path.mkdir()

    @staticmethod
    def load(path):
        path = pathlib.Path(path)
        with open(path / "metadata.json") as f:
            metadata = VcfMetadata.fromdict(json.load(f))
        with open(path / "header.txt") as f:
            header = f.read()
        return PickleChunkedVcf(path, metadata, header)

    @staticmethod
    def convert_partition(
        vcf_metadata,
        partition_index,
        out_path,
        *,
        encoder_threads=4,
        column_chunk_size=16,
    ):
        partition = vcf_metadata.partitions[partition_index]
        vcf = cyvcf2.VCF(partition.vcf_path)
        logger.info(f"Start partition {partition_index} {partition.vcf_path}")

        info_fields = []
        format_fields = []
        has_gt = False
        for field in vcf_metadata.fields:
            if field.category == "INFO":
                info_fields.append(field)
            elif field.category == "FORMAT":
                if field.name == "GT":
                    has_gt = True
                else:
                    format_fields.append(field)

        with ThreadedColumnWriter(
            vcf_metadata,
            out_path,
            partition_index,
            encoder_threads=0,
            chunk_size=column_chunk_size,
        ) as tcw:
            for variant in vcf:
                tcw.append("CHROM", variant.CHROM)
                tcw.append("POS", variant.POS)
                tcw.append("QUAL", variant.QUAL)
                tcw.append("ID", variant.ID)
                tcw.append("FILTERS", variant.FILTERS)
                tcw.append("REF", variant.REF)
                tcw.append("ALT", variant.ALT)
                for field in info_fields:
                    tcw.append(field.full_name, variant.INFO.get(field.name, None))
                if has_gt:
                    tcw.append("FORMAT/GT", variant.genotype.array())
                for field in format_fields:
                    val = None
                    try:
                        val = variant.format(field.name)
                    except KeyError:
                        pass
                    tcw.append(field.full_name, val)

                # Note: an issue with updating the progress per variant here like this
                # is that we get a significant pause at the end of the counter while
                # all the "small" fields get flushed. Possibly not much to be done about it.
                core.update_progress(1)

        return tcw.field_summaries

    @staticmethod
    def convert(
        vcfs, out_path, *, column_chunk_size=16, worker_processes=1, show_progress=False
    ):
        out_path = pathlib.Path(out_path)
        # TODO make scan work in parallel using general progress code too
        vcf_metadata, header = scan_vcfs(vcfs, show_progress=show_progress)
        pcvcf = PickleChunkedVcf(out_path, vcf_metadata, header)
        pcvcf.mkdirs()

        total_variants = sum(
            partition.num_records for partition in vcf_metadata.partitions
        )

        logger.info(
            f"Exploding {pcvcf.num_columns} columns {total_variants} variants "
            f"{pcvcf.num_samples} samples"
        )
        progress_config = core.ProgressConfig(
            total=total_variants, units="vars", title="Explode", show=show_progress
        )
        with core.ParallelWorkManager(worker_processes, progress_config) as pwm:
            for j, partition in enumerate(vcf_metadata.partitions):
                pwm.submit(
                    PickleChunkedVcf.convert_partition,
                    vcf_metadata,
                    j,
                    out_path,
                    column_chunk_size=column_chunk_size,
                )
            partition_summaries = list(pwm.results_as_completed())

        for field in vcf_metadata.fields:
            # Clear the summary to avoid problems when running in debug
            # syncronous mode
            field.summary = VcfFieldSummary()
            for summary in partition_summaries:
                field.summary.update(summary[field.full_name])

        with open(out_path / "metadata.json", "w") as f:
            json.dump(vcf_metadata.asdict(), f, indent=4)
        with open(out_path / "header.txt", "w") as f:
            f.write(header)
        return pcvcf


def explode(
    vcfs,
    out_path,
    *,
    column_chunk_size=16,
    worker_processes=1,
    show_progress=False,
):
    out_path = pathlib.Path(out_path)
    if out_path.exists():
        shutil.rmtree(out_path)

    return PickleChunkedVcf.convert(
        vcfs,
        out_path,
        column_chunk_size=column_chunk_size,
        worker_processes=worker_processes,
        show_progress=show_progress,
    )


def inspect(if_path):
    # TODO add support for the Zarr format also
    pcvcf = PickleChunkedVcf.load(if_path)
    return pcvcf.summary_table()


@dataclasses.dataclass
class ZarrColumnSpec:
    name: str
    dtype: str
    shape: tuple
    chunks: tuple
    dimensions: list
    description: str
    vcf_field: str
    compressor: dict
    # TODO add filters

    def __post_init__(self):
        self.shape = tuple(self.shape)
        self.chunks = tuple(self.chunks)
        self.dimensions = tuple(self.dimensions)


@dataclasses.dataclass
class ZarrConversionSpec:
    chunk_width: int
    chunk_length: int
    dimensions: list
    sample_id: list
    contig_id: list
    contig_length: list
    filter_id: list
    columns: dict

    def asdict(self):
        return dataclasses.asdict(self)

    def asjson(self):
        return json.dumps(self.asdict(), indent=4)

    @staticmethod
    def fromdict(d):
        ret = ZarrConversionSpec(**d)
        ret.columns = {
            key: ZarrColumnSpec(**value) for key, value in d["columns"].items()
        }
        return ret

    @staticmethod
    def fromjson(s):
        return ZarrConversionSpec.fromdict(json.loads(s))

    @staticmethod
    def generate(pcvcf, chunk_length=None, chunk_width=None):
        m = pcvcf.num_records
        n = pcvcf.num_samples
        # FIXME
        if chunk_width is None:
            chunk_width = 1000
        if chunk_length is None:
            chunk_length = 10_000

        compressor = core.default_compressor.get_config()

        def fixed_field_spec(
            name, dtype, vcf_field=None, shape=(m,), dimensions=("variants",)
        ):
            return ZarrColumnSpec(
                vcf_field=vcf_field,
                name=name,
                dtype=dtype,
                shape=shape,
                description="",
                dimensions=dimensions,
                chunks=[chunk_length],
                compressor=compressor,
            )

        alt_col = pcvcf.columns["ALT"]
        max_alleles = alt_col.vcf_field.summary.max_number + 1
        num_filters = len(pcvcf.metadata.filters)

        # # FIXME get dtype from lookup table
        colspecs = [
            fixed_field_spec(
                name="variant_contig",
                dtype="i2",  # FIXME
            ),
            fixed_field_spec(
                name="variant_filter",
                dtype="bool",
                shape=(m, num_filters),
                dimensions=["variants", "filters"],
            ),
            fixed_field_spec(
                name="variant_allele",
                dtype="str",
                shape=[m, max_alleles],
                dimensions=["variants", "alleles"],
            ),
            fixed_field_spec(
                vcf_field="POS",
                name="variant_position",
                dtype="i4",
            ),
            fixed_field_spec(
                vcf_field=None,
                name="variant_id",
                dtype="str",
            ),
            fixed_field_spec(
                vcf_field=None,
                name="variant_id_mask",
                dtype="bool",
            ),
            fixed_field_spec(
                vcf_field="QUAL",
                name="variant_quality",
                dtype="f4",
            ),
        ]

        gt_field = None
        for field in pcvcf.metadata.fields:
            if field.category == "fixed":
                continue
            if field.name == "GT":
                gt_field = field
                continue
            shape = [m]
            prefix = "variant_"
            dimensions = ["variants"]
            chunks = [chunk_length]
            if field.category == "FORMAT":
                prefix = "call_"
                shape.append(n)
                chunks.append(chunk_width),
                dimensions.append("samples")
            # TODO make an option to add in the empty extra dimension
            if field.summary.max_number > 1:
                shape.append(field.summary.max_number)
                dimensions.append(field.name)
            variable_name = prefix + field.name
            colspec = ZarrColumnSpec(
                vcf_field=field.full_name,
                name=variable_name,
                dtype=field.smallest_dtype(),
                shape=shape,
                chunks=chunks,
                dimensions=dimensions,
                description=field.description,
                compressor=compressor,
            )
            colspecs.append(colspec)

        if gt_field is not None:
            ploidy = gt_field.summary.max_number - 1
            shape = [m, n]
            chunks = [chunk_length, chunk_width]
            dimensions = ["variants", "samples"]

            colspecs.append(
                ZarrColumnSpec(
                    vcf_field=None,
                    name="call_genotype_phased",
                    dtype="bool",
                    shape=list(shape),
                    chunks=list(chunks),
                    dimensions=list(dimensions),
                    description="",
                    compressor=compressor,
                )
            )
            shape += [ploidy]
            dimensions += ["ploidy"]
            colspecs.append(
                ZarrColumnSpec(
                    vcf_field=None,
                    name="call_genotype",
                    dtype=gt_field.smallest_dtype(),
                    shape=list(shape),
                    chunks=list(chunks),
                    dimensions=list(dimensions),
                    description="",
                    compressor=compressor,
                )
            )
            colspecs.append(
                ZarrColumnSpec(
                    vcf_field=None,
                    name="call_genotype_mask",
                    dtype="bool",
                    shape=list(shape),
                    chunks=list(chunks),
                    dimensions=list(dimensions),
                    description="",
                    compressor=compressor,
                )
            )

        return ZarrConversionSpec(
            chunk_width=chunk_width,
            chunk_length=chunk_length,
            columns={col.name: col for col in colspecs},
            dimensions=["variants", "samples", "ploidy", "alleles", "filters"],
            sample_id=pcvcf.metadata.samples,
            contig_id=pcvcf.metadata.contig_names,
            contig_length=pcvcf.metadata.contig_lengths,
            filter_id=pcvcf.metadata.filters,
        )


class SgvcfZarr:
    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.root = None

    def create_array(self, variable):
        # print("CREATE", variable)
        object_codec = None
        if variable.dtype == "O":
            object_codec = numcodecs.VLenUTF8()
        a = self.root.empty(
            variable.name,
            shape=variable.shape,
            chunks=variable.chunks,
            dtype=variable.dtype,
            compressor=numcodecs.get_codec(variable.compressor),
            object_codec=object_codec,
        )
        a.attrs["_ARRAY_DIMENSIONS"] = variable.dimensions

    def encode_column(self, pcvcf, column, encoder_threads=4):
        source_col = pcvcf.columns[column.vcf_field]
        array = self.root[column.name]
        ba = core.BufferedArray(array)
        sanitiser = source_col.sanitiser_factory(ba.buff.shape)

        with core.ThreadedZarrEncoder([ba], encoder_threads) as te:
            last_bytes_read = 0
            for value, bytes_read in source_col.iter_values_bytes():
                j = te.next_buffer_row()
                sanitiser(ba.buff, j, value)
                # print(bytes_read, last_bytes_read, value)
                if last_bytes_read != bytes_read:
                    core.update_progress(bytes_read - last_bytes_read)
                    last_bytes_read = bytes_read

    def encode_genotypes(self, pcvcf, encoder_threads=4):
        source_col = pcvcf.columns["FORMAT/GT"]
        gt = core.BufferedArray(self.root["call_genotype"])
        gt_mask = core.BufferedArray(self.root["call_genotype_mask"])
        gt_phased = core.BufferedArray(self.root["call_genotype_phased"])
        buffered_arrays = [gt, gt_phased, gt_mask]

        with core.ThreadedZarrEncoder(buffered_arrays, encoder_threads) as te:
            last_bytes_read = 0
            for value, bytes_read in source_col.iter_values_bytes():
                j = te.next_buffer_row()
                sanitise_value_int_2d(gt.buff, j, value[:, :-1])
                sanitise_value_int_1d(gt_phased.buff, j, value[:, -1])
                # TODO check is this the correct semantics when we are padding
                # with mixed ploidies?
                gt_mask.buff[j] = gt.buff[j] < 0
                if last_bytes_read != bytes_read:
                    core.update_progress(bytes_read - last_bytes_read)
                    last_bytes_read = bytes_read

    def encode_alleles(self, pcvcf):
        ref_col = pcvcf.columns["REF"]
        alt_col = pcvcf.columns["ALT"]
        ref_values = ref_col.values
        alt_values = alt_col.values
        allele_array = self.root["variant_allele"]

        # We could do this chunk-by-chunk, but it doesn't seem worth the bother.
        alleles = np.full(allele_array.shape, "", dtype="O")
        for j, (ref, alt) in enumerate(zip(ref_values, alt_values)):
            alleles[j, 0] = ref[0]
            alleles[j, 1 : 1 + len(alt)] = alt
        allele_array[:] = alleles
        size = sum(
            col.vcf_field.summary.uncompressed_size for col in [ref_col, alt_col]
        )
        core.update_progress(size)
        logger.debug("alleles done")

    def encode_samples(self, pcvcf, sample_id, chunk_width):
        if not np.array_equal(sample_id, pcvcf.metadata.samples):
            raise ValueError("Subsetting or reordering samples not supported currently")
        array = self.root.array(
            "sample_id",
            sample_id,
            dtype="str",
            compressor=core.default_compressor,
            chunks=(chunk_width,),
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ["samples"]
        logger.debug("Samples done")

    def encode_contig(self, pcvcf, contig_names, contig_lengths):
        array = self.root.array(
            "contig_id",
            contig_names,
            dtype="str",
            compressor=core.default_compressor,
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ["contigs"]

        if contig_lengths is not None:
            array = self.root.array(
                "contig_length",
                contig_lengths,
                dtype=np.int64,
            )
            array.attrs["_ARRAY_DIMENSIONS"] = ["contigs"]

        col = pcvcf.columns["CHROM"]
        array = self.root["variant_contig"]
        buff = np.zeros_like(array)
        lookup = {v: j for j, v in enumerate(contig_names)}
        for j, contig in enumerate(col.values):
            try:
                buff[j] = lookup[contig[0]]
            except KeyError:
                # TODO add advice about adding it to the spec
                raise ValueError(f"Contig '{contig}' was not defined in the header.")

        array[:] = buff

        core.update_progress(col.vcf_field.summary.uncompressed_size)
        logger.debug("Contig done")

    def encode_filters(self, pcvcf, filter_names):
        array = self.root.array(
            "filter_id",
            filter_names,
            dtype="str",
            compressor=core.default_compressor,
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ["filters"]

        col = pcvcf.columns["FILTERS"]
        array = self.root["variant_filter"]
        buff = np.zeros_like(array)

        lookup = {v: j for j, v in enumerate(filter_names)}
        for j, filters in enumerate(col.values):
            try:
                for f in filters:
                    buff[j, lookup[f]] = True
            except IndexError:
                raise ValueError(f"Filter '{f}' was not defined in the header.")

        array[:] = buff

        core.update_progress(col.vcf_field.summary.uncompressed_size)
        logger.debug("Filters done")

    def encode_id(self, pcvcf):
        col = pcvcf.columns["ID"]
        id_array = self.root["variant_id"]
        id_mask_array = self.root["variant_id_mask"]
        id_buff = np.full_like(id_array, "")
        id_mask_buff = np.zeros_like(id_mask_array)

        for j, value in enumerate(col.values):
            if value is not None:
                id_buff[j] = value[0]
            else:
                id_buff[j] = "."  # TODO is this correct??
                id_mask_buff[j] = True

        id_array[:] = id_buff
        id_mask_array[:] = id_mask_buff

        core.update_progress(col.vcf_field.summary.uncompressed_size)
        logger.debug("ID done")

    @staticmethod
    def convert(
        pcvcf, path, conversion_spec, *, worker_processes=1, show_progress=False
    ):
        path = pathlib.Path(path)
        # TODO: we should do this as a future to avoid blocking
        if path.exists():
            shutil.rmtree(path)
        write_path = path.with_suffix(path.suffix + f".{os.getpid()}.build")
        store = zarr.DirectoryStore(write_path)
        # FIXME, duplicating logic about the store
        logger.info(f"Create zarr at {write_path}")
        sgvcf = SgvcfZarr(write_path)
        sgvcf.root = zarr.group(store=store, overwrite=True)
        for column in conversion_spec.columns.values():
            sgvcf.create_array(column)

        sgvcf.root.attrs["vcf_zarr_version"] = "0.2"
        sgvcf.root.attrs["vcf_header"] = pcvcf.vcf_header
        sgvcf.root.attrs["source"] = f"bio2zarr-{provenance.__version__}"

        progress_config = core.ProgressConfig(
            total=pcvcf.total_uncompressed_bytes,
            title="Encode",
            units="b",
            show=show_progress,
        )
        with core.ParallelWorkManager(worker_processes, progress_config) as pwm:
            pwm.submit(
                sgvcf.encode_samples,
                pcvcf,
                conversion_spec.sample_id,
                conversion_spec.chunk_width,
            )
            pwm.submit(sgvcf.encode_alleles, pcvcf)
            pwm.submit(sgvcf.encode_id, pcvcf)
            pwm.submit(
                sgvcf.encode_contig,
                pcvcf,
                conversion_spec.contig_id,
                conversion_spec.contig_length,
            )
            pwm.submit(sgvcf.encode_filters, pcvcf, conversion_spec.filter_id)
            has_gt = False
            for variable in conversion_spec.columns.values():
                if variable.vcf_field is not None:
                    # print("Encode", variable.name)
                    # TODO for large columns it's probably worth splitting up
                    # these into vertical chunks. Otherwise we tend to get a
                    # long wait for the largest GT columns to finish.
                    # Straightforward to do because we can chunk-align the work
                    # packages.
                    pwm.submit(sgvcf.encode_column, pcvcf, variable)
                else:
                    if variable.name == "call_genotype":
                        has_gt = True
            if has_gt:
                # TODO add mixed ploidy
                pwm.executor.submit(sgvcf.encode_genotypes, pcvcf)

        zarr.consolidate_metadata(write_path)
        # Atomic swap, now we've completely finished.
        logger.info(f"Moving to final path {path}")
        os.rename(write_path, path)


def mkschema(if_path, out):
    pcvcf = PickleChunkedVcf.load(if_path)
    spec = ZarrConversionSpec.generate(pcvcf)
    out.write(spec.asjson())


def encode(if_path, zarr_path, schema_path, worker_processes=1, show_progress=False):
    pcvcf = PickleChunkedVcf.load(if_path)
    if schema_path is None:
        schema = ZarrConversionSpec.generate(pcvcf)
    else:
        with open(schema_path, "r") as f:
            schema = ZarrConversionSpec.fromjson(f.read())
    SgvcfZarr.convert(
        pcvcf,
        zarr_path,
        conversion_spec=schema,
        worker_processes=worker_processes,
        show_progress=show_progress,
    )


def convert(
    vcfs,
    out_path,
    *,
    chunk_length=None,
    chunk_width=None,
    worker_processes=1,
    show_progress=False,
):
    with tempfile.TemporaryDirectory() as intermediate_form_dir:
        explode(
            vcfs,
            intermediate_form_dir,
            worker_processes=worker_processes,
            show_progress=show_progress,
        )

        pcvcf = PickleChunkedVcf.load(intermediate_form_dir)
        spec = ZarrConversionSpec.generate(
            pcvcf, chunk_length=chunk_length, chunk_width=chunk_width
        )
        SgvcfZarr.convert(
            pcvcf,
            out_path,
            conversion_spec=spec,
            worker_processes=worker_processes,
            show_progress=show_progress,
        )


def assert_all_missing_float(a):
    v = np.array(a, dtype=np.float32).view(np.int32)
    nt.assert_equal(v, FLOAT32_MISSING_AS_INT32)


def assert_all_fill_float(a):
    v = np.array(a, dtype=np.float32).view(np.int32)
    nt.assert_equal(v, FLOAT32_FILL_AS_INT32)


def assert_all_missing_int(a):
    v = np.array(a, dtype=int)
    nt.assert_equal(v, -1)


def assert_all_fill_int(a):
    v = np.array(a, dtype=int)
    nt.assert_equal(v, -2)


def assert_all_missing_string(a):
    nt.assert_equal(a, ".")


def assert_all_fill_string(a):
    nt.assert_equal(a, "")


def assert_all_fill(zarr_val, vcf_type):
    if vcf_type == "Integer":
        assert_all_fill_int(zarr_val)
    elif vcf_type in ("String", "Character"):
        assert_all_fill_string(zarr_val)
    elif vcf_type == "Float":
        assert_all_fill_float(zarr_val)
    else:  # pragma: no cover
        assert False


def assert_all_missing(zarr_val, vcf_type):
    if vcf_type == "Integer":
        assert_all_missing_int(zarr_val)
    elif vcf_type in ("String", "Character"):
        assert_all_missing_string(zarr_val)
    elif vcf_type == "Flag":
        assert zarr_val == False  # noqa 712
    elif vcf_type == "Float":
        assert_all_missing_float(zarr_val)
    else:  # pragma: no cover
        assert False


def assert_info_val_missing(zarr_val, vcf_type):
    assert_all_missing(zarr_val, vcf_type)


def assert_format_val_missing(zarr_val, vcf_type):
    assert_info_val_missing(zarr_val, vcf_type)


# Note: checking exact equality may prove problematic here
# but we should be deterministically storing what cyvcf2
# provides, which should compare equal.


def assert_info_val_equal(vcf_val, zarr_val, vcf_type):
    assert vcf_val is not None
    if vcf_type in ("String", "Character"):
        split = list(vcf_val.split(","))
        k = len(split)
        if isinstance(zarr_val, str):
            assert k == 1
            # Scalar
            assert vcf_val == zarr_val
        else:
            nt.assert_equal(split, zarr_val[:k])
            assert_all_fill(zarr_val[k:], vcf_type)

    elif isinstance(vcf_val, tuple):
        vcf_missing_value_map = {
            "Integer": -1,
            "Float": FLOAT32_MISSING,
        }
        v = [vcf_missing_value_map[vcf_type] if x is None else x for x in vcf_val]
        missing = np.array([j for j, x in enumerate(vcf_val) if x is None], dtype=int)
        a = np.array(v)
        k = len(a)
        # We are checking for int missing twice here, but it's necessary to have
        # a separate check for floats because different NaNs compare equal
        nt.assert_equal(a, zarr_val[:k])
        assert_all_missing(zarr_val[missing], vcf_type)
        if k < len(zarr_val):
            assert_all_fill(zarr_val[k:], vcf_type)
    else:
        # Scalar
        zarr_val = np.array(zarr_val, ndmin=1)
        assert len(zarr_val.shape) == 1
        assert vcf_val == zarr_val[0]
        if len(zarr_val) > 1:
            assert_all_fill(zarr_val[1:], vcf_type)


def assert_format_val_equal(vcf_val, zarr_val, vcf_type):
    assert vcf_val is not None
    assert isinstance(vcf_val, np.ndarray)
    if vcf_type in ("String", "Character"):
        assert len(vcf_val) == len(zarr_val)
        for v, z in zip(vcf_val, zarr_val):
            split = list(v.split(","))
            # Note: deliberately duplicating logic here between this and the
            # INFO col above to make sure all combinations are covered by tests
            k = len(split)
            if k == 1:
                assert v == z
            else:
                nt.assert_equal(split, z[:k])
                assert_all_fill(z[k:], vcf_type)
    else:
        assert vcf_val.shape[0] == zarr_val.shape[0]
        if len(vcf_val.shape) == len(zarr_val.shape) + 1:
            assert vcf_val.shape[-1] == 1
            vcf_val = vcf_val[..., 0]
        assert len(vcf_val.shape) <= 2
        assert len(vcf_val.shape) == len(zarr_val.shape)
        if len(vcf_val.shape) == 2:
            k = vcf_val.shape[1]
            if zarr_val.shape[1] != k:
                assert_all_fill(zarr_val[:, k:], vcf_type)
                zarr_val = zarr_val[:, :k]
        assert vcf_val.shape == zarr_val.shape
        if vcf_type == "Integer":
            vcf_val[vcf_val == VCF_INT_MISSING] = INT_MISSING
            vcf_val[vcf_val == VCF_INT_FILL] = INT_FILL
        elif vcf_type == "Float":
            nt.assert_equal(vcf_val.view(np.int32), zarr_val.view(np.int32))

        nt.assert_equal(vcf_val, zarr_val)


def validate(vcf_path, zarr_path, show_progress=False):
    store = zarr.DirectoryStore(zarr_path)

    root = zarr.group(store=store)
    pos = root["variant_position"][:]
    allele = root["variant_allele"][:]
    chrom = root["contig_id"][:][root["variant_contig"][:]]
    vid = root["variant_id"][:]
    call_genotype = None
    if "call_genotype" in root:
        call_genotype = iter(root["call_genotype"])

    vcf = cyvcf2.VCF(vcf_path)
    format_headers = {}
    info_headers = {}
    for h in vcf.header_iter():
        if h["HeaderType"] == "FORMAT":
            format_headers[h["ID"]] = h
        if h["HeaderType"] == "INFO":
            info_headers[h["ID"]] = h

    format_fields = {}
    info_fields = {}
    for colname in root.keys():
        if colname.startswith("call") and not colname.startswith("call_genotype"):
            vcf_name = colname.split("_", 1)[1]
            vcf_type = format_headers[vcf_name]["Type"]
            format_fields[vcf_name] = vcf_type, iter(root[colname])
        if colname.startswith("variant"):
            name = colname.split("_", 1)[1]
            if name.isupper():
                vcf_type = info_headers[name]["Type"]
                # print(root[colname])
                info_fields[name] = vcf_type, iter(root[colname])
    # print(info_fields)

    first_pos = next(vcf).POS
    start_index = np.searchsorted(pos, first_pos)
    assert pos[start_index] == first_pos
    vcf = cyvcf2.VCF(vcf_path)
    if show_progress:
        iterator = tqdm.tqdm(vcf, total=vcf.num_records)
    else:
        iterator = vcf
    for j, row in enumerate(iterator, start_index):
        assert chrom[j] == row.CHROM
        assert pos[j] == row.POS
        assert vid[j] == ("." if row.ID is None else row.ID)
        assert allele[j, 0] == row.REF
        k = len(row.ALT)
        nt.assert_array_equal(allele[j, 1 : k + 1], row.ALT),
        assert np.all(allele[j, k + 1 :] == "")
        # TODO FILTERS

        if call_genotype is None:
            val = None
            try:
                val = row.format("GT")
            except KeyError:
                pass
            assert val is None
        else:
            gt = row.genotype.array()
            gt_zarr = next(call_genotype)
            gt_vcf = gt[:, :-1]
            # NOTE cyvcf2 remaps genotypes automatically
            # into the same missing/pad encoding that sgkit uses.
            nt.assert_array_equal(gt_zarr, gt_vcf)

        for name, (vcf_type, zarr_iter) in info_fields.items():
            vcf_val = row.INFO.get(name, None)
            zarr_val = next(zarr_iter)
            if vcf_val is None:
                assert_info_val_missing(zarr_val, vcf_type)
            else:
                assert_info_val_equal(vcf_val, zarr_val, vcf_type)

        for name, (vcf_type, zarr_iter) in format_fields.items():
            vcf_val = None
            try:
                vcf_val = row.format(name)
            except KeyError:
                pass
            zarr_val = next(zarr_iter)
            if vcf_val is None:
                assert_format_val_missing(zarr_val, vcf_type)
            else:
                assert_format_val_equal(vcf_val, zarr_val, vcf_type)
