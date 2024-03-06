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
from typing import Any, List

import humanfriendly
import cyvcf2
import numcodecs
import numpy as np
import numpy.testing as nt
import tqdm
import zarr

from . import core
from . import provenance
from . import vcf_utils

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
    region: str
    num_records: int = -1


@dataclasses.dataclass
class VcfMetadata:
    format_version: str
    samples: list
    contig_names: list
    contig_record_counts: dict
    filters: list
    fields: list
    partitions: list = None
    contig_lengths: list = None

    @property
    def info_fields(self):
        fields = []
        for field in self.fields:
            if field.category == "INFO":
                fields.append(field)
        return fields

    @property
    def format_fields(self):
        fields = []
        for field in self.fields:
            if field.category == "FORMAT":
                fields.append(field)
        return fields

    @property
    def num_records(self):
        return sum(self.contig_record_counts.values())

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


def scan_vcf(path, target_num_partitions):
    with vcf_utils.IndexedVcf(path) as indexed_vcf:
        vcf = indexed_vcf.vcf
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
            contig_record_counts=indexed_vcf.contig_record_counts(),
            filters=filters,
            # TODO use the mapping dictionary
            fields=fields,
            partitions=[],
            # FIXME do something systematic with this
            format_version="0.1",
        )
        try:
            metadata.contig_lengths = vcf.seqlens
        except AttributeError:
            pass

        regions = indexed_vcf.partition_into_regions(num_parts=target_num_partitions)
        logger.info(
            f"Split {path} into {len(regions)} regions (target={target_num_partitions})"
        )
        for region in regions:
            metadata.partitions.append(
                VcfPartition(
                    vcf_path=str(path),
                    region=region,
                )
            )
        core.update_progress(1)
        return metadata, vcf.raw_header


def scan_vcfs(paths, show_progress, target_num_partitions, worker_processes=1):
    logger.info(f"Scanning {len(paths)} VCFs")
    progress_config = core.ProgressConfig(
        total=len(paths),
        units="files",
        title="Scan",
        show=show_progress,
    )
    with core.ParallelWorkManager(worker_processes, progress_config) as pwm:
        for path in paths:
            pwm.submit(scan_vcf, path, target_num_partitions)
        results = list(pwm.results_as_completed())

    # Sort to make the ordering deterministic
    results.sort(key=lambda t: t[0].partitions[0].vcf_path)
    # We just take the first header, assuming the others
    # are compatible.
    all_partitions = []
    contig_record_counts = collections.Counter()
    for metadata, _ in results:
        all_partitions.extend(metadata.partitions)
        metadata.partitions.clear()
        contig_record_counts += metadata.contig_record_counts
        metadata.contig_record_counts.clear()

    vcf_metadata, header = results[0]
    for metadata, _ in results[1:]:
        if metadata != vcf_metadata:
            raise ValueError("Incompatible VCF chunks")

    vcf_metadata.contig_record_counts = dict(contig_record_counts)

    # Sort by contig (in the order they appear in the header) first,
    # then by start coordinate
    contig_index_map = {contig: j for j, contig in enumerate(metadata.contig_names)}
    all_partitions.sort(
        key=lambda x: (contig_index_map[x.region.contig], x.region.start)
    )
    vcf_metadata.partitions = all_partitions
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
    def __init__(self, pcvcf, vcf_field):
        self.vcf_field = vcf_field
        self.path = self.get_path(pcvcf.path, vcf_field)
        self.compressor = pcvcf.compressor
        self.num_partitions = pcvcf.num_partitions
        self.num_records = pcvcf.num_records
        self.partition_record_index = pcvcf.partition_record_index
        # A map of partition id to the cumulative number of records
        # in chunks within that partition
        self._chunk_record_index = {}

    @staticmethod
    def get_path(base_path, vcf_field):
        if vcf_field.category == "fixed":
            return base_path / vcf_field.name
        return base_path / vcf_field.category / vcf_field.name

    @property
    def name(self):
        return self.vcf_field.full_name

    def partition_path(self, partition_id):
        return self.path / f"p{partition_id}"

    def __repr__(self):
        partition_chunks = [self.num_chunks(j) for j in range(self.num_partitions)]
        return (
            f"PickleChunkedVcfField(name={self.name}, "
            f"partition_chunks={partition_chunks}, "
            f"path={self.path})"
        )

    def num_chunks(self, partition_id):
        return len(self.chunk_cumulative_records(partition_id))

    def chunk_record_index(self, partition_id):
        if partition_id not in self._chunk_record_index:
            index_path = self.partition_path(partition_id) / "chunk_index.pkl"
            with open(index_path, "rb") as f:
                a = pickle.load(f)
            assert len(a) > 1
            assert a[0] == 0
            self._chunk_record_index[partition_id] = a
        return self._chunk_record_index[partition_id]

    def chunk_cumulative_records(self, partition_id):
        return self.chunk_record_index(partition_id)[1:]

    def chunk_num_records(self, partition_id):
        return np.diff(self.chunk_cumulative_records(partition_id))

    def chunk_files(self, partition_id, start=0):
        partition_path = self.partition_path(partition_id)
        for n in self.chunk_cumulative_records(partition_id)[start:]:
            yield partition_path / f"{n}.pkl"

    def read_chunk(self, path):
        with open(path, "rb") as f:
            pkl = self.compressor.decode(f.read())
        return pickle.loads(pkl)

    def iter_values(self, start=None, stop=None):
        start = 0 if start is None else start
        stop = self.num_records if stop is None else stop
        start_partition = (
            np.searchsorted(self.partition_record_index, start, side="right") - 1
        )
        offset = self.partition_record_index[start_partition]
        assert offset <= start
        chunk_offset = start - offset

        chunk_record_index = self.chunk_record_index(start_partition)
        start_chunk = (
            np.searchsorted(chunk_record_index, chunk_offset, side="right") - 1
        )
        record_id = offset + chunk_record_index[start_chunk]
        assert record_id <= start
        logger.debug(
            f"Read {self.vcf_field.full_name} slice [{start}:{stop}]:"
            f"p_start={start_partition}, c_start={start_chunk}, r_start={record_id}"
        )

        for chunk_path in self.chunk_files(start_partition, start_chunk):
            chunk = self.read_chunk(chunk_path)
            for record in chunk:
                if record_id == stop:
                    return
                if record_id >= start:
                    yield record
                record_id += 1
        assert record_id > start
        for partition_id in range(start_partition + 1, self.num_partitions):
            for chunk_path in self.chunk_files(partition_id):
                chunk = self.read_chunk(chunk_path)
                for record in chunk:
                    if record_id == stop:
                        return
                    yield record
                    record_id += 1

    # Note: this involves some computation so should arguably be a method,
    # but making a property for consistency with xarray etc
    @property
    def values(self):
        ret = [None] * self.num_records
        j = 0
        for partition_id in range(self.num_partitions):
            for chunk_path in self.chunk_files(partition_id):
                chunk = self.read_chunk(chunk_path)
                for record in chunk:
                    ret[j] = record
                    j += 1
        if j != self.num_records:
            raise ValueError(
                f"Corruption detected: incorrect number of records in {str(self.path)}."
            )
        return ret

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
class PcvcfFieldWriter:
    vcf_field: VcfField
    path: pathlib.Path
    transformer: VcfValueTransformer
    compressor: Any
    max_buffered_bytes: int
    buff: List[Any] = dataclasses.field(default_factory=list)
    buffered_bytes: int = 0
    chunk_index: List[int] = dataclasses.field(default_factory=lambda: [0])
    num_records: int = 0

    def append(self, val):
        val = self.transformer.transform_and_update_bounds(val)
        assert val is None or isinstance(val, np.ndarray)
        self.buff.append(val)
        val_bytes = sys.getsizeof(val)
        self.buffered_bytes += val_bytes
        self.num_records += 1
        if self.buffered_bytes >= self.max_buffered_bytes:
            logger.debug(
                f"Flush {self.path} buffered={self.buffered_bytes} "
                f"max={self.max_buffered_bytes}"
            )
            self.write_chunk()
            self.buff.clear()
            self.buffered_bytes = 0

    def write_chunk(self):
        # Update index
        self.chunk_index.append(self.num_records)
        path = self.path / f"{self.num_records}.pkl"
        logger.debug(f"Start write: {path}")
        pkl = pickle.dumps(self.buff)
        compressed = self.compressor.encode(pkl)
        with open(path, "wb") as f:
            f.write(compressed)

        # Update the summary
        self.vcf_field.summary.num_chunks += 1
        self.vcf_field.summary.compressed_size += len(compressed)
        self.vcf_field.summary.uncompressed_size += self.buffered_bytes
        logger.debug(f"Finish write: {path}")

    def flush(self):
        logger.debug(
            f"Flush {self.path} records={len(self.buff)} buffered={self.buffered_bytes}"
        )
        if len(self.buff) > 0:
            self.write_chunk()
        with open(self.path / "chunk_index.pkl", "wb") as f:
            a = np.array(self.chunk_index, dtype=int)
            pickle.dump(a, f)


class PcvcfPartitionWriter(contextlib.AbstractContextManager):
    """
    Writes the data for a PickleChunkedVcf partition.
    """

    def __init__(
        self,
        vcf_metadata,
        out_path,
        partition_index,
        compressor,
        *,
        chunk_size=1,
    ):
        self.partition_index = partition_index
        # chunk_size is in megabytes
        max_buffered_bytes = chunk_size * 2**20
        assert max_buffered_bytes > 0

        self.field_writers = {}
        num_samples = len(vcf_metadata.samples)
        for vcf_field in vcf_metadata.fields:
            field_path = PickleChunkedVcfField.get_path(out_path, vcf_field)
            field_partition_path = field_path / f"p{partition_index}"
            transformer = VcfValueTransformer.factory(vcf_field, num_samples)
            self.field_writers[vcf_field.full_name] = PcvcfFieldWriter(
                vcf_field,
                field_partition_path,
                transformer,
                compressor,
                max_buffered_bytes,
            )

    @property
    def field_summaries(self):
        return {
            name: field.vcf_field.summary for name, field in self.field_writers.items()
        }

    def append(self, name, value):
        self.field_writers[name].append(value)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            for field in self.field_writers.values():
                field.flush()
        return False


class PickleChunkedVcf(collections.abc.Mapping):
    # TODO Check if other compressors would give reasonable compression
    # with significantly faster times
    DEFAULT_COMPRESSOR = numcodecs.Blosc(cname="zstd", clevel=7)

    def __init__(self, path, metadata, vcf_header):
        self.path = path
        self.metadata = metadata
        self.vcf_header = vcf_header
        self.compressor = self.DEFAULT_COMPRESSOR
        self.columns = {}
        partition_num_records = [
            partition.num_records for partition in self.metadata.partitions
        ]
        # Allow us to find which partition a given record is in
        self.partition_record_index = np.cumsum([0] + partition_num_records)
        for field in self.metadata.fields:
            self.columns[field.full_name] = PickleChunkedVcfField(self, field)

    def __repr__(self):
        return (
            f"PickleChunkedVcf(fields={len(self)}, partitions={self.num_partitions}, "
            f"records={self.num_records}, path={self.path})"
        )

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
        return sum(self.metadata.contig_record_counts.values())

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
        pcvcf = PickleChunkedVcf(path, metadata, header)
        logger.info(
            f"Loaded PickleChunkedVcf(partitions={pcvcf.num_partitions}, "
            f"records={pcvcf.num_records}, columns={pcvcf.num_columns})"
        )
        return pcvcf

    @staticmethod
    def convert_partition(
        vcf_metadata,
        partition_index,
        out_path,
        *,
        column_chunk_size=16,
    ):
        partition = vcf_metadata.partitions[partition_index]
        logger.info(
            f"Start p{partition_index} {partition.vcf_path}__{partition.region}"
        )
        info_fields = vcf_metadata.info_fields
        format_fields = []
        has_gt = False
        for field in vcf_metadata.format_fields:
            if field.name == "GT":
                has_gt = True
            else:
                format_fields.append(field)

        compressor = PickleChunkedVcf.DEFAULT_COMPRESSOR

        with PcvcfPartitionWriter(
            vcf_metadata,
            out_path,
            partition_index,
            compressor,
            chunk_size=column_chunk_size,
        ) as tcw:
            with vcf_utils.IndexedVcf(partition.vcf_path) as ivcf:
                num_records = 0
                for variant in ivcf.variants(partition.region):
                    num_records += 1
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

        logger.info(
            f"Finish p{partition_index} {partition.vcf_path}__{partition.region}="
            f"{num_records} records"
        )
        return partition_index, tcw.field_summaries, num_records

    @staticmethod
    def convert(
        vcfs, out_path, *, column_chunk_size=16, worker_processes=1, show_progress=False
    ):
        out_path = pathlib.Path(out_path)
        # TODO make scan work in parallel using general progress code too
        target_num_partitions = max(1, worker_processes * 4)
        vcf_metadata, header = scan_vcfs(
            vcfs,
            worker_processes=worker_processes,
            show_progress=show_progress,
            target_num_partitions=target_num_partitions,
        )
        pcvcf = PickleChunkedVcf(out_path, vcf_metadata, header)
        pcvcf.mkdirs()

        logger.info(
            f"Exploding {pcvcf.num_columns} columns {vcf_metadata.num_records} variants "
            f"{pcvcf.num_samples} samples"
        )
        progress_config = core.ProgressConfig(
            total=vcf_metadata.num_records,
            units="vars",
            title="Explode",
            show=show_progress,
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
            num_records = 0
            partition_summaries = []
            for index, summary, num_records in pwm.results_as_completed():
                partition_summaries.append(summary)
                vcf_metadata.partitions[index].num_records = num_records

        total_records = sum(
            partition.num_records for partition in vcf_metadata.partitions
        )
        assert total_records == pcvcf.num_records

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

    PickleChunkedVcf.convert(
        vcfs,
        out_path,
        column_chunk_size=column_chunk_size,
        worker_processes=worker_processes,
        show_progress=show_progress,
    )
    return PickleChunkedVcf.load(out_path)


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
    format_version: str
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
        logger.info(f"Generating schema with chunks={chunk_length, chunk_width}")
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
            # TODO do something systematic
            format_version="0.1",
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

    def encode_column_slice(self, pcvcf, column, start, stop):
        source_col = pcvcf.columns[column.vcf_field]
        array = self.root[column.name]
        ba = core.BufferedArray(array, start)
        sanitiser = source_col.sanitiser_factory(ba.buff.shape)

        for value in source_col.iter_values(start, stop):
            # We write directly into the buffer in the sanitiser function
            # to make it easier to reason about dimension padding
            j = ba.next_buffer_row()
            sanitiser(ba.buff, j, value)
        ba.flush()
        logger.debug(f"Encoded {column.name} slice {start}:{stop}")

    def encode_genotypes_slice(self, pcvcf, start, stop):
        source_col = pcvcf.columns["FORMAT/GT"]
        gt = core.BufferedArray(self.root["call_genotype"], start)
        gt_mask = core.BufferedArray(self.root["call_genotype_mask"], start)
        gt_phased = core.BufferedArray(self.root["call_genotype_phased"], start)

        for value in source_col.iter_values(start, stop):
            j = gt.next_buffer_row()
            sanitise_value_int_2d(gt.buff, j, value[:, :-1])
            j = gt_phased.next_buffer_row()
            sanitise_value_int_1d(gt_phased.buff, j, value[:, -1])
            # TODO check is this the correct semantics when we are padding
            # with mixed ploidies?
            j = gt_mask.next_buffer_row()
            gt_mask.buff[j] = gt.buff[j] < 0
        gt.flush()
        gt_phased.flush()
        gt_mask.flush()
        logger.debug(f"Encoded GT slice {start}:{stop}")

    def encode_alleles_slice(self, pcvcf, start, stop):
        ref_col = pcvcf.columns["REF"]
        alt_col = pcvcf.columns["ALT"]
        alleles = core.BufferedArray(self.root["variant_allele"], start)

        for ref, alt in zip(
            ref_col.iter_values(start, stop), alt_col.iter_values(start, stop)
        ):
            j = alleles.next_buffer_row()
            alleles.buff[j, :] = STR_FILL
            alleles.buff[j, 0] = ref[0]
            alleles.buff[j, 1 : 1 + len(alt)] = alt
        alleles.flush()
        logger.debug(f"Encoded alleles slice {start}:{stop}")

    def encode_id_slice(self, pcvcf, start, stop):
        col = pcvcf.columns["ID"]
        vid = core.BufferedArray(self.root["variant_id"], start)
        vid_mask = core.BufferedArray(self.root["variant_id_mask"], start)

        for value in col.iter_values(start, stop):
            j = vid.next_buffer_row()
            k = vid_mask.next_buffer_row()
            assert j == k
            if value is not None:
                vid.buff[j] = value[0]
                vid_mask.buff[j] = False
            else:
                vid.buff[j] = STR_MISSING
                vid_mask.buff[j] = True
        vid.flush()
        vid_mask.flush()
        logger.debug(f"Encoded ID slice {start}:{stop}")

    def encode_filters_slice(self, pcvcf, lookup, start, stop):
        col = pcvcf.columns["FILTERS"]
        var_filter = core.BufferedArray(self.root["variant_filter"], start)

        for value in col.iter_values(start, stop):
            j = var_filter.next_buffer_row()
            var_filter.buff[j] = False
            try:
                for f in value:
                    var_filter.buff[j, lookup[f]] = True
            except IndexError:
                raise ValueError(f"Filter '{f}' was not defined in the header.")
        var_filter.flush()
        logger.debug(f"Encoded FILTERS slice {start}:{stop}")

    def encode_contig_slice(self, pcvcf, lookup, start, stop):
        col = pcvcf.columns["CHROM"]
        contig = core.BufferedArray(self.root["variant_contig"], start)

        for value in col.iter_values(start, stop):
            j = contig.next_buffer_row()
            try:
                contig.buff[j] = lookup[value[0]]
            except KeyError:
                # TODO add advice about adding it to the spec
                raise ValueError(f"Contig '{contig}' was not defined in the header.")
        contig.flush()
        logger.debug(f"Encoded CHROM slice {start}:{stop}")

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

    def encode_contig_id(self, pcvcf, contig_names, contig_lengths):
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
        return {v: j for j, v in enumerate(contig_names)}

    def encode_filter_id(self, pcvcf, filter_names):
        array = self.root.array(
            "filter_id",
            filter_names,
            dtype="str",
            compressor=core.default_compressor,
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ["filters"]
        return {v: j for j, v in enumerate(filter_names)}

    @staticmethod
    def encode(
        pcvcf,
        path,
        conversion_spec,
        *,
        worker_processes=1,
        max_v_chunks=None,
        show_progress=False,
    ):
        path = pathlib.Path(path)
        # TODO: we should do this as a future to avoid blocking
        if path.exists():
            logger.warning(f"Deleting existing {path}")
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

        num_slices = max(1, worker_processes * 4)
        # Using POS arbitrarily to get the array slices
        slices = core.chunk_aligned_slices(
            sgvcf.root["variant_position"], num_slices, max_chunks=max_v_chunks
        )
        truncated = slices[-1][-1]
        for array in sgvcf.root.values():
            if array.attrs["_ARRAY_DIMENSIONS"][0] == "variants":
                shape = list(array.shape)
                shape[0] = truncated
                array.resize(shape)

        chunked_1d = [
            col for col in conversion_spec.columns.values() if len(col.chunks) <= 1
        ]
        progress_config = core.ProgressConfig(
            total=sum(sgvcf.root[col.name].nchunks for col in chunked_1d),
            title="Encode 1D",
            units="chunks",
            show=show_progress,
        )

        # Do these syncronously for simplicity so we have the mapping
        filter_id_map = sgvcf.encode_filter_id(pcvcf, conversion_spec.filter_id)
        contig_id_map = sgvcf.encode_contig_id(
            pcvcf, conversion_spec.contig_id, conversion_spec.contig_length
        )

        with core.ParallelWorkManager(worker_processes, progress_config) as pwm:
            pwm.submit(
                sgvcf.encode_samples,
                pcvcf,
                conversion_spec.sample_id,
                conversion_spec.chunk_width,
            )
            for start, stop in slices:
                pwm.submit(sgvcf.encode_alleles_slice, pcvcf, start, stop)
                pwm.submit(sgvcf.encode_id_slice, pcvcf, start, stop)
                pwm.submit(
                    sgvcf.encode_filters_slice, pcvcf, filter_id_map, start, stop
                )
                pwm.submit(sgvcf.encode_contig_slice, pcvcf, contig_id_map, start, stop)
                for col in chunked_1d:
                    if col.vcf_field is not None:
                        pwm.submit(sgvcf.encode_column_slice, pcvcf, col, start, stop)

        chunked_2d = [
            col for col in conversion_spec.columns.values() if len(col.chunks) >= 2
        ]
        if len(chunked_2d) > 0:
            progress_config = core.ProgressConfig(
                total=sum(sgvcf.root[col.name].nchunks for col in chunked_2d),
                title="Encode 2D",
                units="chunks",
                show=show_progress,
            )
            with core.ParallelWorkManager(worker_processes, progress_config) as pwm:
                if "call_genotype" in conversion_spec.columns:
                    logger.info(f"Submit encode call_genotypes in {len(slices)} slices")
                    for start, stop in slices:
                        pwm.submit(sgvcf.encode_genotypes_slice, pcvcf, start, stop)

                for col in chunked_2d:
                    if col.vcf_field is not None:
                        logger.info(f"Submit encode {col.name} in {len(slices)} slices")
                        for start, stop in slices:
                            pwm.submit(
                                sgvcf.encode_column_slice, pcvcf, col, start, stop
                            )

        zarr.consolidate_metadata(write_path)
        # Atomic swap, now we've completely finished.
        logger.info(f"Moving to final path {path}")
        os.rename(write_path, path)


def mkschema(if_path, out):
    pcvcf = PickleChunkedVcf.load(if_path)
    spec = ZarrConversionSpec.generate(pcvcf)
    out.write(spec.asjson())


def encode(
    if_path,
    zarr_path,
    schema_path=None,
    chunk_length=None,
    chunk_width=None,
    max_v_chunks=None,
    worker_processes=1,
    show_progress=False,
):
    pcvcf = PickleChunkedVcf.load(if_path)
    if schema_path is None:
        schema = ZarrConversionSpec.generate(
            pcvcf,
            chunk_length=chunk_length,
            chunk_width=chunk_width,
        )
    else:
        logger.info(f"Reading schema from {schema_path}")
        if chunk_length is not None or chunk_width is not None:
            raise ValueError("Cannot specify schema along with chunk sizes")
        with open(schema_path, "r") as f:
            schema = ZarrConversionSpec.fromjson(f.read())

    SgvcfZarr.encode(
        pcvcf,
        zarr_path,
        conversion_spec=schema,
        max_v_chunks=max_v_chunks,
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
    # TODO add arguments to control location of tmpdir
):
    with tempfile.TemporaryDirectory(prefix="vcf2zarr_if_") as if_dir:
        explode(
            vcfs,
            if_dir,
            worker_processes=worker_processes,
            show_progress=show_progress,
        )
        encode(
            if_dir,
            out_path,
            chunk_length=chunk_length,
            chunk_width=chunk_width,
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


# TODO rename to "verify"
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
        iterator = tqdm.tqdm(vcf, desc="   Verify", total=vcf.num_records)
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
