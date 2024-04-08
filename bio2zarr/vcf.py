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


def display_number(x):
    ret = "n/a"
    if math.isfinite(x):
        ret = f"{x: 0.2g}"
    return ret


def display_size(n):
    return humanfriendly.format_size(n, binary=True)


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
        self.uncompressed_size += other.uncompressed_size
        self.max_number = max(self.max_number, other.max_number)
        self.min_value = min(self.min_value, other.min_value)
        self.max_value = max(self.max_value, other.max_value)

    def asdict(self):
        return dataclasses.asdict(self)

    @staticmethod
    def fromdict(d):
        return VcfFieldSummary(**d)


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


ICF_METADATA_FORMAT_VERSION = "0.2"
ICF_DEFAULT_COMPRESSOR = numcodecs.Blosc(
    cname="zstd", clevel=7, shuffle=numcodecs.Blosc.NOSHUFFLE
)


@dataclasses.dataclass
class IcfMetadata:
    samples: list
    contig_names: list
    contig_record_counts: dict
    filters: list
    fields: list
    partitions: list = None
    contig_lengths: list = None
    format_version: str = None
    compressor: dict = None
    column_chunk_size: int = None
    provenance: dict = None

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
        if d["format_version"] != ICF_METADATA_FORMAT_VERSION:
            raise ValueError(
                "Intermediate columnar metadata format version mismatch: "
                f"{d['format_version']} != {ICF_METADATA_FORMAT_VERSION}"
            )
        fields = [VcfField.fromdict(fd) for fd in d["fields"]]
        partitions = [VcfPartition(**pd) for pd in d["partitions"]]
        for p in partitions:
            p.region = vcf_utils.Region(**p.region)
        d = d.copy()
        d["fields"] = fields
        d["partitions"] = partitions
        return IcfMetadata(**d)

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

        metadata = IcfMetadata(
            samples=vcf.samples,
            contig_names=vcf.seqnames,
            contig_record_counts=indexed_vcf.contig_record_counts(),
            filters=filters,
            fields=fields,
            partitions=[],
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
                    # TODO should this be fully resolving the path? Otherwise it's all
                    # relative to the original WD
                    vcf_path=str(path),
                    region=region,
                )
            )
        core.update_progress(1)
        return metadata, vcf.raw_header


def scan_vcfs(paths, show_progress, target_num_partitions, worker_processes=1):
    logger.info(
        f"Scanning {len(paths)} VCFs attempting to split into {target_num_partitions} partitions."
    )
    # An easy mistake to make is to pass the same file twice. Check this early on.
    for path, count in collections.Counter(paths).items():
        if not path.exists():  # NEEDS TEST
            raise FileNotFoundError(path)
        if count > 1:
            raise ValueError(f"Duplicate path provided: {path}")

    progress_config = core.ProgressConfig(
        total=len(paths),
        units="files",
        title="Scan",
        show=show_progress,
    )
    with core.ParallelWorkManager(worker_processes, progress_config) as pwm:
        for path in paths:
            pwm.submit(scan_vcf, path, max(1, target_num_partitions // len(paths)))
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

    icf_metadata, header = results[0]
    for metadata, _ in results[1:]:
        if metadata != icf_metadata:
            raise ValueError("Incompatible VCF chunks")

    icf_metadata.contig_record_counts = dict(contig_record_counts)

    # Sort by contig (in the order they appear in the header) first,
    # then by start coordinate
    contig_index_map = {contig: j for j, contig in enumerate(metadata.contig_names)}
    all_partitions.sort(
        key=lambda x: (contig_index_map[x.region.contig], x.region.start)
    )
    icf_metadata.partitions = all_partitions
    logger.info(f"Scan complete, resulting in {len(all_partitions)} partitions.")
    return icf_metadata, header


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
        value = [VCF_INT_MISSING if x is None else x for x in value]  #  NEEDS TEST
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
    in the IntermediateColumnarFormat, and update field summaries.
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
            return self.missing_value  # NEEDS TEST
        assert self.dimension == 1
        return np.array(vcf_value, ndmin=1, dtype="str")


def get_vcf_field_path(base_path, vcf_field):
    if vcf_field.category == "fixed":
        return base_path / vcf_field.name
    return base_path / vcf_field.category / vcf_field.name


class IntermediateColumnarFormatField:
    def __init__(self, icf, vcf_field):
        self.vcf_field = vcf_field
        self.path = get_vcf_field_path(icf.path, vcf_field)
        self.compressor = icf.compressor
        self.num_partitions = icf.num_partitions
        self.num_records = icf.num_records
        self.partition_record_index = icf.partition_record_index
        # A map of partition id to the cumulative number of records
        # in chunks within that partition
        self._chunk_record_index = {}

    @property
    def name(self):
        return self.vcf_field.full_name

    def partition_path(self, partition_id):
        return self.path / f"p{partition_id}"

    def __repr__(self):
        partition_chunks = [self.num_chunks(j) for j in range(self.num_partitions)]
        return (
            f"IntermediateColumnarFormatField(name={self.name}, "
            f"partition_chunks={partition_chunks}, "
            f"path={self.path})"
        )

    def num_chunks(self, partition_id):
        return len(self.chunk_record_index(partition_id)) - 1

    def chunk_record_index(self, partition_id):
        if partition_id not in self._chunk_record_index:
            index_path = self.partition_path(partition_id) / "chunk_index"
            with open(index_path, "rb") as f:
                a = pickle.load(f)
            assert len(a) > 1
            assert a[0] == 0
            self._chunk_record_index[partition_id] = a
        return self._chunk_record_index[partition_id]

    def read_chunk(self, path):
        with open(path, "rb") as f:
            pkl = self.compressor.decode(f.read())
        return pickle.loads(pkl)

    def chunk_num_records(self, partition_id):
        return np.diff(self.chunk_record_index(partition_id))

    def chunks(self, partition_id, start_chunk=0):
        partition_path = self.partition_path(partition_id)
        chunk_cumulative_records = self.chunk_record_index(partition_id)
        chunk_num_records = np.diff(chunk_cumulative_records)
        for count, cumulative in zip(
            chunk_num_records[start_chunk:], chunk_cumulative_records[start_chunk + 1 :]
        ):
            path = partition_path / f"{cumulative}"
            chunk = self.read_chunk(path)
            if len(chunk) != count:
                raise ValueError(f"Corruption detected in chunk: {path}")
            yield chunk

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
        for chunk in self.chunks(start_partition, start_chunk):
            for record in chunk:
                if record_id == stop:
                    return
                if record_id >= start:
                    yield record
                record_id += 1
        assert record_id > start
        for partition_id in range(start_partition + 1, self.num_partitions):
            for chunk in self.chunks(partition_id):
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
            for chunk in self.chunks(partition_id):
                for record in chunk:
                    ret[j] = record
                    j += 1
        assert j == self.num_records
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
class IcfFieldWriter:
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
        path = self.path / f"{self.num_records}"
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
        with open(self.path / "chunk_index", "wb") as f:
            a = np.array(self.chunk_index, dtype=int)
            pickle.dump(a, f)


class IcfPartitionWriter(contextlib.AbstractContextManager):
    """
    Writes the data for a IntermediateColumnarFormat partition.
    """

    def __init__(
        self,
        icf_metadata,
        out_path,
        partition_index,
    ):
        self.partition_index = partition_index
        # chunk_size is in megabytes
        max_buffered_bytes = icf_metadata.column_chunk_size * 2**20
        assert max_buffered_bytes > 0
        compressor = numcodecs.get_codec(icf_metadata.compressor)

        self.field_writers = {}
        num_samples = len(icf_metadata.samples)
        for vcf_field in icf_metadata.fields:
            field_path = get_vcf_field_path(out_path, vcf_field)
            field_partition_path = field_path / f"p{partition_index}"
            transformer = VcfValueTransformer.factory(vcf_field, num_samples)
            self.field_writers[vcf_field.full_name] = IcfFieldWriter(
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


class IntermediateColumnarFormat(collections.abc.Mapping):
    def __init__(self, path):
        self.path = pathlib.Path(path)
        # TODO raise a more informative error here telling people this
        # directory is either a WIP or the wrong format.
        with open(self.path / "metadata.json") as f:
            self.metadata = IcfMetadata.fromdict(json.load(f))
        with open(self.path / "header.txt") as f:
            self.vcf_header = f.read()

        self.compressor = numcodecs.get_codec(self.metadata.compressor)
        self.columns = {}
        partition_num_records = [
            partition.num_records for partition in self.metadata.partitions
        ]
        # Allow us to find which partition a given record is in
        self.partition_record_index = np.cumsum([0] + partition_num_records)
        for field in self.metadata.fields:
            self.columns[field.full_name] = IntermediateColumnarFormatField(self, field)
        logger.info(
            f"Loaded IntermediateColumnarFormat(partitions={self.num_partitions}, "
            f"records={self.num_records}, columns={self.num_columns})"
        )

    def __repr__(self):
        return (
            f"IntermediateColumnarFormat(fields={len(self)}, partitions={self.num_partitions}, "
            f"records={self.num_records}, path={self.path})"
        )

    def __getitem__(self, key):
        return self.columns[key]

    def __iter__(self):
        return iter(self.columns)

    def __len__(self):
        return len(self.columns)

    def summary_table(self):
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


class IntermediateColumnarFormatWriter:
    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.wip_path = self.path / "wip"
        self.metadata = None

    @property
    def num_partitions(self):
        return len(self.metadata.partitions)

    def init(
        self,
        vcfs,
        *,
        column_chunk_size=16,
        worker_processes=1,
        target_num_partitions=None,
        show_progress=False,
        compressor=None,
    ):
        if self.path.exists():
            raise ValueError("ICF path already exists")
        if compressor is None:
            compressor = ICF_DEFAULT_COMPRESSOR
        vcfs = [pathlib.Path(vcf) for vcf in vcfs]
        target_num_partitions = max(target_num_partitions, len(vcfs))

        # TODO move scan_vcfs into this class
        icf_metadata, header = scan_vcfs(
            vcfs,
            worker_processes=worker_processes,
            show_progress=show_progress,
            target_num_partitions=target_num_partitions,
        )
        self.metadata = icf_metadata
        self.metadata.format_version = ICF_METADATA_FORMAT_VERSION
        self.metadata.compressor = compressor.get_config()
        self.metadata.column_chunk_size = column_chunk_size
        # Bare minimum here for provenance - would be nice to include versions of key
        # dependencies as well.
        self.metadata.provenance = {"source": f"bio2zarr-{provenance.__version__}"}

        self.mkdirs(worker_processes)

        # Note: this is needed for the current version of the vcfzarr spec, but it's
        # probably going to be dropped.
        # https://github.com/pystatgen/vcf-zarr-spec/issues/15
        # May be useful to keep lying around still though?
        logger.info(f"Writing VCF header")
        with open(self.path / "header.txt", "w") as f:
            f.write(header)

        logger.info(f"Writing WIP metadata")
        with open(self.wip_path / "metadata.json", "w") as f:
            json.dump(self.metadata.asdict(), f, indent=4)
        return self.num_partitions

    def mkdirs(self, worker_processes=1):
        logger.info(
            f"Creating {len(self.metadata.fields) * self.num_partitions} directories"
        )
        self.path.mkdir()
        self.wip_path.mkdir()
        # Due to high latency batch system filesystems, we create all the directories in
        # parallel
        progress_config = core.ProgressConfig(
            total=len(self.metadata.fields) * self.num_partitions,
            units="dir",
            title="Creating directories",
            show=True
        )
        with core.ParallelWorkManager(
                worker_processes=worker_processes,
                progress_config=progress_config
        ) as manager:
            for field in self.metadata.fields:
                col_path = get_vcf_field_path(self.path, field)
                manager.submit(col_path.mkdir, parents=True)
                for j in range(self.num_partitions):
                    part_path = col_path / f"p{j}"
                    manager.submit(part_path.mkdir, parents=True)

    def load_partition_summaries(self):
        summaries = []
        not_found = []
        for j in range(self.num_partitions):
            try:
                with open(self.wip_path / f"p{j}_summary.json") as f:
                    summary = json.load(f)
                    for k, v in summary["field_summaries"].items():
                        summary["field_summaries"][k] = VcfFieldSummary.fromdict(v)
                    summaries.append(summary)
            except FileNotFoundError:
                not_found.append(j)
        if len(not_found) > 0:
            raise FileNotFoundError(
                f"Partition metadata not found for {len(not_found)} partitions: {not_found}"
            )
        return summaries

    def load_metadata(self):
        if self.metadata is None:
            with open(self.wip_path / f"metadata.json") as f:
                self.metadata = IcfMetadata.fromdict(json.load(f))

    def process_partition(self, partition_index):
        self.load_metadata()
        summary_path = self.wip_path / f"p{partition_index}_summary.json"
        # If someone is rewriting a summary path (for whatever reason), make sure it
        # doesn't look like it's already been completed.
        # NOTE to do this properly we probably need to take a lock on this file - but
        # this simple approach will catch the vast majority of problems.
        if summary_path.exists():
            summary_path.unlink()

        partition = self.metadata.partitions[partition_index]
        logger.info(
            f"Start p{partition_index} {partition.vcf_path}__{partition.region}"
        )
        info_fields = self.metadata.info_fields
        format_fields = []
        has_gt = False
        for field in self.metadata.format_fields:
            if field.name == "GT":
                has_gt = True
            else:
                format_fields.append(field)

        with IcfPartitionWriter(
            self.metadata,
            self.path,
            partition_index,
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
                        val = variant.format(field.name)
                        tcw.append(field.full_name, val)
                    # Note: an issue with updating the progress per variant here like this
                    # is that we get a significant pause at the end of the counter while
                    # all the "small" fields get flushed. Possibly not much to be done about it.
                    core.update_progress(1)
            logger.info(
                f"Finished reading VCF for partition {partition_index}, flushing buffers"
            )

        partition_metadata = {
            "num_records": num_records,
            "field_summaries": {k: v.asdict() for k, v in tcw.field_summaries.items()},
        }
        with open(summary_path, "w") as f:
            json.dump(partition_metadata, f, indent=4)
        logger.info(
            f"Finish p{partition_index} {partition.vcf_path}__{partition.region}="
            f"{num_records} records"
        )

    def process_partition_slice(
        self,
        start,
        stop,
        *,
        worker_processes=1,
        show_progress=False,
    ):
        self.load_metadata()
        if start == 0 and stop == self.num_partitions:
            num_records = self.metadata.num_records
        else:
            # We only know the number of records if all partitions are done at once,
            # and we signal this to tqdm by passing None as the total.
            num_records = None
        num_columns = len(self.metadata.fields)
        num_samples = len(self.metadata.samples)
        logger.info(
            f"Exploding columns={num_columns} samples={num_samples}; "
            f"partitions={stop - start} "
            f"variants={'unknown' if num_records is None else num_records}"
        )
        progress_config = core.ProgressConfig(
            total=num_records,
            units="vars",
            title="Explode",
            show=show_progress,
        )
        with core.ParallelWorkManager(worker_processes, progress_config) as pwm:
            for j in range(start, stop):
                pwm.submit(self.process_partition, j)

    def explode(self, *, worker_processes=1, show_progress=False):
        self.load_metadata()
        return self.process_partition_slice(
            0,
            self.num_partitions,
            worker_processes=worker_processes,
            show_progress=show_progress,
        )

    def explode_partition(self, partition, *, show_progress=False, worker_processes=1):
        self.load_metadata()
        if partition < 0 or partition >= self.num_partitions:
            raise ValueError(
                "Partition index must be in the range 0 <= index < num_partitions"
            )
        return self.process_partition_slice(
            partition,
            partition + 1,
            worker_processes=worker_processes,
            show_progress=show_progress,
        )

    def finalise(self):
        self.load_metadata()
        partition_summaries = self.load_partition_summaries()
        total_records = 0
        for index, summary in enumerate(partition_summaries):
            partition_records = summary["num_records"]
            self.metadata.partitions[index].num_records = partition_records
            total_records += partition_records
        assert total_records == self.metadata.num_records

        for field in self.metadata.fields:
            for summary in partition_summaries:
                field.summary.update(summary["field_summaries"][field.full_name])

        logger.info(f"Finalising metadata")
        with open(self.path / "metadata.json", "w") as f:
            json.dump(self.metadata.asdict(), f, indent=4)

        logger.debug(f"Removing WIP directory")
        shutil.rmtree(self.wip_path)


def explode(
    icf_path,
    vcfs,
    *,
    column_chunk_size=16,
    worker_processes=1,
    show_progress=False,
    compressor=None,
):
    writer = IntermediateColumnarFormatWriter(icf_path)
    num_partitions = writer.init(
        vcfs,
        # Heuristic to get reasonable worker utilisation with lumpy partition sizing
        target_num_partitions=max(1, worker_processes * 4),
        worker_processes=worker_processes,
        show_progress=show_progress,
        column_chunk_size=column_chunk_size,
        compressor=compressor,
    )
    writer.explode(worker_processes=worker_processes, show_progress=show_progress)
    writer.finalise()
    return IntermediateColumnarFormat(icf_path)


def explode_init(
    icf_path,
    vcfs,
    *,
    column_chunk_size=16,
    target_num_partitions=1,
    worker_processes=1,
    show_progress=False,
    compressor=None,
):
    writer = IntermediateColumnarFormatWriter(icf_path)
    return writer.init(
        vcfs,
        target_num_partitions=target_num_partitions,
        worker_processes=worker_processes,
        show_progress=show_progress,
        column_chunk_size=column_chunk_size,
        compressor=compressor,
    )


# NOTE only including worker_processes here so we can use the 0 option to get the
# work done syncronously and so we can get test coverage on it. Should find a
# better way to do this.
def explode_partition(icf_path, partition, *, show_progress=False, worker_processes=1):
    writer = IntermediateColumnarFormatWriter(icf_path)
    writer.explode_partition(
        partition, show_progress=show_progress, worker_processes=worker_processes
    )


def explode_finalise(icf_path):
    writer = IntermediateColumnarFormatWriter(icf_path)
    writer.finalise()


def inspect(path):
    path = pathlib.Path(path)
    # TODO add support for the Zarr format also
    if (path / "metadata.json").exists():
        obj = IntermediateColumnarFormat(path)
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
    dimensions: list
    description: str
    vcf_field: str
    compressor: dict = None
    filters: list = None
    # TODO add filters

    def __post_init__(self):
        self.shape = tuple(self.shape)
        self.chunks = tuple(self.chunks)
        self.dimensions = tuple(self.dimensions)
        self.compressor = DEFAULT_ZARR_COMPRESSOR.get_config()
        self.filters = []
        self._choose_compressor_settings()

    def _choose_compressor_settings(self):
        """
        Choose compressor and filter settings based on the size and
        type of the array, plus some hueristics from observed properties
        of VCFs.

        See https://github.com/pystatgen/bio2zarr/discussions/74
        """
        dt = np.dtype(self.dtype)
        # Default is to not shuffle, because autoshuffle isn't recognised
        # by many Zarr implementations, and shuffling can lead to worse
        # performance in some cases anyway. Turning on shuffle should be a
        # deliberate choice.
        shuffle = numcodecs.Blosc.NOSHUFFLE
        if dt.itemsize == 1:
            # Any 1 byte field gets BITSHUFFLE by default
            shuffle = numcodecs.Blosc.BITSHUFFLE
        self.compressor["shuffle"] = shuffle


ZARR_SCHEMA_FORMAT_VERSION = "0.2"


@dataclasses.dataclass
class VcfZarrSchema:
    format_version: str
    samples_chunk_size: int
    variants_chunk_size: int
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
        if d["format_version"] != ZARR_SCHEMA_FORMAT_VERSION:
            raise ValueError(
                "Zarr schema format version mismatch: "
                f"{d['format_version']} != {ZARR_SCHEMA_FORMAT_VERSION}"
            )
        ret = VcfZarrSchema(**d)
        ret.columns = {
            key: ZarrColumnSpec(**value) for key, value in d["columns"].items()
        }
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
                chunks=[variants_chunk_size],
            )

        alt_col = icf.columns["ALT"]
        max_alleles = alt_col.vcf_field.summary.max_number + 1
        num_filters = len(icf.metadata.filters)

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
        for field in icf.metadata.fields:
            if field.category == "fixed":
                continue
            if field.name == "GT":
                gt_field = field
                continue
            shape = [m]
            prefix = "variant_"
            dimensions = ["variants"]
            chunks = [variants_chunk_size]
            if field.category == "FORMAT":
                prefix = "call_"
                shape.append(n)
                chunks.append(samples_chunk_size),
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
            )
            colspecs.append(colspec)

        if gt_field is not None:
            ploidy = gt_field.summary.max_number - 1
            shape = [m, n]
            chunks = [variants_chunk_size, samples_chunk_size]
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
                )
            )

        return VcfZarrSchema(
            format_version=ZARR_SCHEMA_FORMAT_VERSION,
            samples_chunk_size=samples_chunk_size,
            variants_chunk_size=variants_chunk_size,
            columns={col.name: col for col in colspecs},
            dimensions=["variants", "samples", "ploidy", "alleles", "filters"],
            sample_id=icf.metadata.samples,
            contig_id=icf.metadata.contig_names,
            contig_length=icf.metadata.contig_lengths,
            filter_id=icf.metadata.filters,
        )


class VcfZarr:
    def __init__(self, path):
        if not (path / ".zmetadata").exists():
            raise ValueError("Not in VcfZarr format")  # NEEDS TEST
        self.root = zarr.open(path, mode="r")

    def __repr__(self):
        return repr(self.root)  # NEEDS TEST

    def summary_table(self):
        data = []
        arrays = [(a.nbytes_stored, a) for _, a in self.root.arrays()]
        arrays.sort(key=lambda x: x[0])
        for stored, array in reversed(arrays):
            d = {
                "name": array.name,
                "dtype": str(array.dtype),
                "stored": display_size(stored),
                "size": display_size(array.nbytes),
                "ratio": display_number(array.nbytes / stored),
                "nchunks": str(array.nchunks),
                "chunk_size": display_size(array.nbytes / array.nchunks),
                "avg_chunk_stored": display_size(int(stored / array.nchunks)),
                "shape": str(array.shape),
                "chunk_shape": str(array.chunks),
                "compressor": str(array.compressor),
                "filters": str(array.filters),
            }
            data.append(d)
        return data


@dataclasses.dataclass
class EncodingWork:
    func: callable = dataclasses.field(repr=False)
    start: int
    stop: int
    columns: list[str]
    memory: int = 0


def parse_max_memory(max_memory):
    if max_memory is None:
        # Effectively unbounded
        return 2**63
    if isinstance(max_memory, str):
        max_memory = humanfriendly.parse_size(max_memory)
    logger.info(f"Set memory budget to {display_size(max_memory)}")
    return max_memory


class VcfZarrWriter:
    def __init__(self, path, icf, schema, dimension_separator=None):
        self.path = pathlib.Path(path)
        self.icf = icf
        self.schema = schema
        store = zarr.DirectoryStore(self.path)
        # Default to using nested directories following the Zarr v3 default.
        self.dimension_separator = (
            "/" if dimension_separator is None else dimension_separator
        )
        self.root = zarr.group(store=store)

    def init_array(self, variable):
        # print("CREATE", variable)
        object_codec = None
        if variable.dtype == "O":
            object_codec = numcodecs.VLenUTF8()
        a = self.root.empty(
            "wip_" + variable.name,
            shape=variable.shape,
            chunks=variable.chunks,
            dtype=variable.dtype,
            compressor=numcodecs.get_codec(variable.compressor),
            filters=[numcodecs.get_codec(filt) for filt in variable.filters],
            object_codec=object_codec,
            dimension_separator=self.dimension_separator,
        )
        # Dimension names are part of the spec in Zarr v3
        a.attrs["_ARRAY_DIMENSIONS"] = variable.dimensions

    def get_array(self, name):
        return self.root["wip_" + name]

    def finalise_array(self, variable_name):
        source = self.path / ("wip_" + variable_name)
        dest = self.path / variable_name
        # Atomic swap
        os.rename(source, dest)
        logger.info(f"Finalised {variable_name}")

    def encode_array_slice(self, column, start, stop):
        source_col = self.icf.columns[column.vcf_field]
        array = self.get_array(column.name)
        ba = core.BufferedArray(array, start)
        sanitiser = source_col.sanitiser_factory(ba.buff.shape)

        for value in source_col.iter_values(start, stop):
            # We write directly into the buffer in the sanitiser function
            # to make it easier to reason about dimension padding
            j = ba.next_buffer_row()
            sanitiser(ba.buff, j, value)
        ba.flush()
        logger.debug(f"Encoded {column.name} slice {start}:{stop}")

    def encode_genotypes_slice(self, start, stop):
        source_col = self.icf.columns["FORMAT/GT"]
        gt = core.BufferedArray(self.get_array("call_genotype"), start)
        gt_mask = core.BufferedArray(self.get_array("call_genotype_mask"), start)
        gt_phased = core.BufferedArray(self.get_array("call_genotype_phased"), start)

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

    def encode_alleles_slice(self, start, stop):
        ref_col = self.icf.columns["REF"]
        alt_col = self.icf.columns["ALT"]
        alleles = core.BufferedArray(self.get_array("variant_allele"), start)

        for ref, alt in zip(
            ref_col.iter_values(start, stop), alt_col.iter_values(start, stop)
        ):
            j = alleles.next_buffer_row()
            alleles.buff[j, :] = STR_FILL
            alleles.buff[j, 0] = ref[0]
            alleles.buff[j, 1 : 1 + len(alt)] = alt
        alleles.flush()
        logger.debug(f"Encoded alleles slice {start}:{stop}")

    def encode_id_slice(self, start, stop):
        col = self.icf.columns["ID"]
        vid = core.BufferedArray(self.get_array("variant_id"), start)
        vid_mask = core.BufferedArray(self.get_array("variant_id_mask"), start)

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

    def encode_filters_slice(self, lookup, start, stop):
        col = self.icf.columns["FILTERS"]
        var_filter = core.BufferedArray(self.get_array("variant_filter"), start)

        for value in col.iter_values(start, stop):
            j = var_filter.next_buffer_row()
            var_filter.buff[j] = False
            for f in value:
                try:
                    var_filter.buff[j, lookup[f]] = True
                except KeyError:
                    raise ValueError(f"Filter '{f}' was not defined in the header.")
        var_filter.flush()
        logger.debug(f"Encoded FILTERS slice {start}:{stop}")

    def encode_contig_slice(self, lookup, start, stop):
        col = self.icf.columns["CHROM"]
        contig = core.BufferedArray(self.get_array("variant_contig"), start)

        for value in col.iter_values(start, stop):
            j = contig.next_buffer_row()
            # Note: because we are using the indexes to define the lookups
            # and we always have an index, it seems that we the contig lookup
            # will always succeed. However, if anyone ever does hit a KeyError
            # here, please do open an issue with a reproducible example!
            contig.buff[j] = lookup[value[0]]
        contig.flush()
        logger.debug(f"Encoded CHROM slice {start}:{stop}")

    def encode_samples(self):
        if not np.array_equal(self.schema.sample_id, self.icf.metadata.samples):
            raise ValueError(
                "Subsetting or reordering samples not supported currently"
            )  # NEEDS TEST
        array = self.root.array(
            "sample_id",
            self.schema.sample_id,
            dtype="str",
            compressor=DEFAULT_ZARR_COMPRESSOR,
            chunks=(self.schema.samples_chunk_size,),
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ["samples"]
        logger.debug("Samples done")

    def encode_contig_id(self):
        array = self.root.array(
            "contig_id",
            self.schema.contig_id,
            dtype="str",
            compressor=DEFAULT_ZARR_COMPRESSOR,
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ["contigs"]
        if self.schema.contig_length is not None:
            array = self.root.array(
                "contig_length",
                self.schema.contig_length,
                dtype=np.int64,
                compressor=DEFAULT_ZARR_COMPRESSOR,
            )
            array.attrs["_ARRAY_DIMENSIONS"] = ["contigs"]
        return {v: j for j, v in enumerate(self.schema.contig_id)}

    def encode_filter_id(self):
        array = self.root.array(
            "filter_id",
            self.schema.filter_id,
            dtype="str",
            compressor=DEFAULT_ZARR_COMPRESSOR,
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ["filters"]
        return {v: j for j, v in enumerate(self.schema.filter_id)}

    def init(self):
        self.root.attrs["vcf_zarr_version"] = "0.2"
        self.root.attrs["vcf_header"] = self.icf.vcf_header
        self.root.attrs["source"] = f"bio2zarr-{provenance.__version__}"
        for column in self.schema.columns.values():
            self.init_array(column)

    def finalise(self):
        zarr.consolidate_metadata(self.path)

    def encode(
        self,
        worker_processes=1,
        max_v_chunks=None,
        show_progress=False,
        max_memory=None,
    ):
        max_memory = parse_max_memory(max_memory)

        # TODO this will move into the setup logic later when we're making it possible
        # to split the work by slice
        num_slices = max(1, worker_processes * 4)
        # Using POS arbitrarily to get the array slices
        slices = core.chunk_aligned_slices(
            self.get_array("variant_position"), num_slices, max_chunks=max_v_chunks
        )
        truncated = slices[-1][-1]
        for array in self.root.values():
            if array.attrs["_ARRAY_DIMENSIONS"][0] == "variants":
                shape = list(array.shape)
                shape[0] = truncated
                array.resize(shape)

        total_bytes = 0
        encoding_memory_requirements = {}
        for col in self.schema.columns.values():
            array = self.get_array(col.name)
            # NOTE!! this is bad, we're potentially creating quite a large
            # numpy array for basically nothing. We can compute this.
            variant_chunk_size = array.blocks[0].nbytes
            encoding_memory_requirements[col.name] = variant_chunk_size
            logger.debug(
                f"{col.name} requires at least {display_size(variant_chunk_size)} per worker"
            )
            total_bytes += array.nbytes

        filter_id_map = self.encode_filter_id()
        contig_id_map = self.encode_contig_id()

        work = []
        for start, stop in slices:
            for col in self.schema.columns.values():
                if col.vcf_field is not None:
                    f = functools.partial(self.encode_array_slice, col)
                    work.append(
                        EncodingWork(
                            f,
                            start,
                            stop,
                            [col.name],
                            encoding_memory_requirements[col.name],
                        )
                    )
            work.append(
                EncodingWork(self.encode_alleles_slice, start, stop, ["variant_allele"])
            )
            work.append(
                EncodingWork(
                    self.encode_id_slice, start, stop, ["variant_id", "variant_id_mask"]
                )
            )
            work.append(
                EncodingWork(
                    functools.partial(self.encode_filters_slice, filter_id_map),
                    start,
                    stop,
                    ["variant_filter"],
                )
            )
            work.append(
                EncodingWork(
                    functools.partial(self.encode_contig_slice, contig_id_map),
                    start,
                    stop,
                    ["variant_contig"],
                )
            )
            if "call_genotype" in self.schema.columns:
                variables = [
                    "call_genotype",
                    "call_genotype_phased",
                    "call_genotype_mask",
                ]
                gt_memory = sum(
                    encoding_memory_requirements[name] for name in variables
                )
                work.append(
                    EncodingWork(
                        self.encode_genotypes_slice, start, stop, variables, gt_memory
                    )
                )

        # Fail early if we can't fit a particular column into memory
        for wp in work:
            if wp.memory > max_memory:
                raise ValueError(
                    f"Insufficient memory for {wp.columns}: "
                    f"{display_size(wp.memory)} > {display_size(max_memory)}"
                )

        progress_config = core.ProgressConfig(
            total=total_bytes,
            title="Encode",
            units="B",
            show=show_progress,
        )

        used_memory = 0
        # We need to keep some bounds on the queue size or the memory bounds algorithm
        # below doesn't really work.
        max_queued = 4 * max(1, worker_processes)
        encoded_slices = collections.Counter()

        with core.ParallelWorkManager(worker_processes, progress_config) as pwm:
            future = pwm.submit(self.encode_samples)
            future_to_work = {future: EncodingWork(None, 0, 0, [])}

            def service_completed_futures():
                nonlocal used_memory

                completed = pwm.wait_for_completed()
                for future in completed:
                    wp_done = future_to_work.pop(future)
                    used_memory -= wp_done.memory
                    logger.debug(
                        f"Complete {wp_done}: used mem={display_size(used_memory)}"
                    )
                    for column in wp_done.columns:
                        encoded_slices[column] += 1
                        if encoded_slices[column] == len(slices):
                            # Do this syncronously for simplicity. Should be
                            # fine as the workers will probably be busy with
                            # large encode tasks most of the time.
                            self.finalise_array(column)

            for wp in work:
                while (
                    used_memory + wp.memory > max_memory
                    or len(future_to_work) > max_queued
                ):
                    logger.debug(
                        f"Wait: mem_required={used_memory + wp.memory} max_mem={max_memory} "
                        f"queued={len(future_to_work)} max_queued={max_queued}"
                    )
                    service_completed_futures()
                future = pwm.submit(wp.func, wp.start, wp.stop)
                used_memory += wp.memory
                logger.debug(f"Submit {wp}: used mem={display_size(used_memory)}")
                future_to_work[future] = wp

            logger.debug("All work submitted")
            while len(future_to_work) > 0:
                service_completed_futures()


def mkschema(if_path, out):
    icf = IntermediateColumnarFormat(if_path)
    spec = VcfZarrSchema.generate(icf)
    out.write(spec.asjson())


def encode(
    if_path,
    zarr_path,
    schema_path=None,
    variants_chunk_size=None,
    samples_chunk_size=None,
    max_v_chunks=None,
    max_memory=None,
    worker_processes=1,
    show_progress=False,
):
    icf = IntermediateColumnarFormat(if_path)
    if schema_path is None:
        schema = VcfZarrSchema.generate(
            icf,
            variants_chunk_size=variants_chunk_size,
            samples_chunk_size=samples_chunk_size,
        )
    else:
        logger.info(f"Reading schema from {schema_path}")
        if variants_chunk_size is not None or samples_chunk_size is not None:
            raise ValueError(
                "Cannot specify schema along with chunk sizes"
            )  # NEEDS TEST
        with open(schema_path, "r") as f:
            schema = VcfZarrSchema.fromjson(f.read())
    zarr_path = pathlib.Path(zarr_path)
    if zarr_path.exists():
        logger.warning(f"Deleting existing {zarr_path}")
        shutil.rmtree(zarr_path)
    vzw = VcfZarrWriter(zarr_path, icf, schema)
    vzw.init()
    vzw.encode(
        max_v_chunks=max_v_chunks,
        worker_processes=worker_processes,
        max_memory=max_memory,
        show_progress=show_progress,
    )
    vzw.finalise()


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
        if_dir = pathlib.Path(tmp) / "if"
        explode(
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
                info_fields[name] = vcf_type, iter(root[colname])

    first_pos = next(vcf).POS
    start_index = np.searchsorted(pos, first_pos)
    assert pos[start_index] == first_pos
    vcf = cyvcf2.VCF(vcf_path)
    if show_progress:
        iterator = tqdm.tqdm(vcf, desc=" Verify", total=vcf.num_records)  # NEEDS TEST
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
            vcf_val = row.format(name)
            zarr_val = next(zarr_iter)
            if vcf_val is None:
                assert_format_val_missing(zarr_val, vcf_type)
            else:
                assert_format_val_equal(vcf_val, zarr_val, vcf_type)
