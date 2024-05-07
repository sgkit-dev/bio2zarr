import collections
import contextlib
import dataclasses
import json
import logging
import math
import os
import os.path
import pathlib
import pickle
import shutil
import sys
import tempfile
from typing import Any

import cyvcf2
import humanfriendly
import numcodecs
import numpy as np
import numpy.testing as nt
import tqdm
import zarr

from . import core, provenance, vcf_utils

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

    def smallest_dtype(self):
        """
        Returns the smallest dtype suitable for this field based
        on type, and values.
        """
        s = self.summary
        if self.vcf_type == "Float":
            ret = "f4"
        elif self.vcf_type == "Integer":
            if not math.isfinite(s.max_value):
                # All missing values; use i1. Note we should have some API to
                # check more explicitly for missingness:
                # https://github.com/sgkit-dev/bio2zarr/issues/131
                ret = "i1"
            else:
                ret = core.min_int_dtype(s.min_value, s.max_value)
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


ICF_METADATA_FORMAT_VERSION = "0.3"
ICF_DEFAULT_COMPRESSOR = numcodecs.Blosc(
    cname="zstd", clevel=7, shuffle=numcodecs.Blosc.NOSHUFFLE
)


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
class IcfMetadata:
    samples: list
    contigs: list
    filters: list
    fields: list
    partitions: list = None
    format_version: str = None
    compressor: dict = None
    column_chunk_size: int = None
    provenance: dict = None
    num_records: int = -1

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
    def num_contigs(self):
        return len(self.contigs)

    @property
    def num_filters(self):
        return len(self.filters)

    @property
    def num_samples(self):
        return len(self.samples)

    @staticmethod
    def fromdict(d):
        if d["format_version"] != ICF_METADATA_FORMAT_VERSION:
            raise ValueError(
                "Intermediate columnar metadata format version mismatch: "
                f"{d['format_version']} != {ICF_METADATA_FORMAT_VERSION}"
            )
        partitions = [VcfPartition(**pd) for pd in d["partitions"]]
        for p in partitions:
            p.region = vcf_utils.Region(**p.region)
        d = d.copy()
        d["partitions"] = partitions
        d["fields"] = [VcfField.fromdict(fd) for fd in d["fields"]]
        d["samples"] = [Sample(**sd) for sd in d["samples"]]
        d["filters"] = [Filter(**fd) for fd in d["filters"]]
        d["contigs"] = [Contig(**cd) for cd in d["contigs"]]
        return IcfMetadata(**d)

    def asdict(self):
        return dataclasses.asdict(self)

    def asjson(self):
        return json.dumps(self.asdict(), indent=4)


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
        filters = []
        pass_index = -1
        for h in vcf.header_iter():
            if h["HeaderType"] == "FILTER" and isinstance(h["ID"], str):
                try:
                    description = h["Description"].strip('"')
                except KeyError:
                    description = ""
                if h["ID"] == "PASS":
                    pass_index = len(filters)
                filters.append(Filter(h["ID"], description))

        # Ensure PASS is the first filter if present
        if pass_index > 0:
            pass_filter = filters.pop(pass_index)
            filters.insert(0, pass_filter)

        fields = fixed_vcf_field_definitions()
        for h in vcf.header_iter():
            if h["HeaderType"] in ["INFO", "FORMAT"]:
                field = VcfField.from_header(h)
                if field.name == "GT":
                    field.vcf_type = "Integer"
                    field.vcf_number = "."
                fields.append(field)

        try:
            contig_lengths = vcf.seqlens
        except AttributeError:
            contig_lengths = [None for _ in vcf.seqnames]

        metadata = IcfMetadata(
            samples=[Sample(sample_id) for sample_id in vcf.samples],
            contigs=[
                Contig(contig_id, length)
                for contig_id, length in zip(vcf.seqnames, contig_lengths)
            ],
            filters=filters,
            fields=fields,
            partitions=[],
            num_records=sum(indexed_vcf.contig_record_counts().values()),
        )

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
        f"Scanning {len(paths)} VCFs attempting to split into {target_num_partitions}"
        f" partitions."
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
    total_records = 0
    for metadata, _ in results:
        for partition in metadata.partitions:
            logger.debug(f"Scanned partition {partition}")
            all_partitions.append(partition)
        total_records += metadata.num_records
        metadata.num_records = 0
        metadata.partitions = []

    icf_metadata, header = results[0]
    for metadata, _ in results[1:]:
        if metadata != icf_metadata:
            raise ValueError("Incompatible VCF chunks")

    # Note: this will be infinity here if any of the chunks has an index
    # that doesn't keep track of the number of records per-contig
    icf_metadata.num_records = total_records

    # Sort by contig (in the order they appear in the header) first,
    # then by start coordinate
    contig_index_map = {contig.id: j for j, contig in enumerate(metadata.contigs)}
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
        x = [FLOAT32_MISSING]
    buff[j] = x[0]


def sanitise_value_int_scalar(buff, j, value):
    x = value
    if value is None:
        # print("MISSING", INT_MISSING, INT_FILL)
        x = [INT_MISSING]
    else:
        x = sanitise_int_array(value, ndmin=1, dtype=np.int32)
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
        value = [VCF_INT_MISSING if x is None else x for x in value]  # NEEDS TEST
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
    buff: list[Any] = dataclasses.field(default_factory=list)
    buffered_bytes: int = 0
    chunk_index: list[int] = dataclasses.field(default_factory=lambda: [0])
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
            # Should be robust to running explode_partition twice.
            field_partition_path.mkdir(exist_ok=True)
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
        self.fields = {}
        partition_num_records = [
            partition.num_records for partition in self.metadata.partitions
        ]
        # Allow us to find which partition a given record is in
        self.partition_record_index = np.cumsum([0, *partition_num_records])
        for field in self.metadata.fields:
            self.fields[field.full_name] = IntermediateColumnarFormatField(self, field)
        logger.info(
            f"Loaded IntermediateColumnarFormat(partitions={self.num_partitions}, "
            f"records={self.num_records}, fields={self.num_fields})"
        )

    def __repr__(self):
        return (
            f"IntermediateColumnarFormat(fields={len(self)}, "
            f"partitions={self.num_partitions}, "
            f"records={self.num_records}, path={self.path})"
        )

    def __getitem__(self, key):
        return self.fields[key]

    def __iter__(self):
        return iter(self.fields)

    def __len__(self):
        return len(self.fields)

    def summary_table(self):
        data = []
        for name, col in self.fields.items():
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

    @property
    def num_records(self):
        return self.metadata.num_records

    @property
    def num_partitions(self):
        return len(self.metadata.partitions)

    @property
    def num_samples(self):
        return len(self.metadata.samples)

    @property
    def num_fields(self):
        return len(self.fields)


@dataclasses.dataclass
class IcfPartitionMetadata:
    num_records: int
    last_position: int
    field_summaries: dict

    def asdict(self):
        return dataclasses.asdict(self)

    def asjson(self):
        return json.dumps(self.asdict(), indent=4)

    @staticmethod
    def fromdict(d):
        md = IcfPartitionMetadata(**d)
        for k, v in md.field_summaries.items():
            md.field_summaries[k] = VcfFieldSummary.fromdict(v)
        return md


def check_overlapping_partitions(partitions):
    for i in range(1, len(partitions)):
        prev_region = partitions[i - 1].region
        current_region = partitions[i].region
        if prev_region.contig == current_region.contig:
            assert prev_region.end is not None
            # Regions are *inclusive*
            if prev_region.end >= current_region.start:
                raise ValueError(
                    f"Overlapping VCF regions in partitions {i - 1} and {i}: "
                    f"{prev_region} and {current_region}"
                )


def check_field_clobbering(icf_metadata):
    info_field_names = set(field.name for field in icf_metadata.info_fields)
    fixed_variant_fields = set(
        ["contig", "id", "id_mask", "position", "allele", "filter", "quality"]
    )
    intersection = info_field_names & fixed_variant_fields
    if len(intersection) > 0:
        raise ValueError(
            f"INFO field name(s) clashing with VCF Zarr spec: {intersection}"
        )

    format_field_names = set(field.name for field in icf_metadata.format_fields)
    fixed_variant_fields = set(["genotype", "genotype_phased", "genotype_mask"])
    intersection = format_field_names & fixed_variant_fields
    if len(intersection) > 0:
        raise ValueError(
            f"FORMAT field name(s) clashing with VCF Zarr spec: {intersection}"
        )


@dataclasses.dataclass
class IcfWriteSummary:
    num_partitions: int
    num_samples: int
    num_variants: int

    def asdict(self):
        return dataclasses.asdict(self)

    def asjson(self):
        return json.dumps(self.asdict(), indent=4)


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
        check_field_clobbering(icf_metadata)
        self.metadata = icf_metadata
        self.metadata.format_version = ICF_METADATA_FORMAT_VERSION
        self.metadata.compressor = compressor.get_config()
        self.metadata.column_chunk_size = column_chunk_size
        # Bare minimum here for provenance - would be nice to include versions of key
        # dependencies as well.
        self.metadata.provenance = {"source": f"bio2zarr-{provenance.__version__}"}

        self.mkdirs()

        # Note: this is needed for the current version of the vcfzarr spec, but it's
        # probably going to be dropped.
        # https://github.com/pystatgen/vcf-zarr-spec/issues/15
        # May be useful to keep lying around still though?
        logger.info("Writing VCF header")
        with open(self.path / "header.txt", "w") as f:
            f.write(header)

        logger.info("Writing WIP metadata")
        with open(self.wip_path / "metadata.json", "w") as f:
            json.dump(self.metadata.asdict(), f, indent=4)
        return IcfWriteSummary(
            num_partitions=self.num_partitions,
            num_variants=icf_metadata.num_records,
            num_samples=icf_metadata.num_samples,
        )

    def mkdirs(self):
        num_dirs = len(self.metadata.fields)
        logger.info(f"Creating {num_dirs} field directories")
        self.path.mkdir()
        self.wip_path.mkdir()
        for field in self.metadata.fields:
            col_path = get_vcf_field_path(self.path, field)
            col_path.mkdir(parents=True)

    def load_partition_summaries(self):
        summaries = []
        not_found = []
        for j in range(self.num_partitions):
            try:
                with open(self.wip_path / f"p{j}.json") as f:
                    summaries.append(IcfPartitionMetadata.fromdict(json.load(f)))
            except FileNotFoundError:
                not_found.append(j)
        if len(not_found) > 0:
            raise FileNotFoundError(
                f"Partition metadata not found for {len(not_found)}"
                f" partitions: {not_found}"
            )
        return summaries

    def load_metadata(self):
        if self.metadata is None:
            with open(self.wip_path / "metadata.json") as f:
                self.metadata = IcfMetadata.fromdict(json.load(f))

    def process_partition(self, partition_index):
        self.load_metadata()
        summary_path = self.wip_path / f"p{partition_index}.json"
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

        last_position = None
        with IcfPartitionWriter(
            self.metadata,
            self.path,
            partition_index,
        ) as tcw:
            with vcf_utils.IndexedVcf(partition.vcf_path) as ivcf:
                num_records = 0
                for variant in ivcf.variants(partition.region):
                    num_records += 1
                    last_position = variant.POS
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
                    # Note: an issue with updating the progress per variant here like
                    # this is that we get a significant pause at the end of the counter
                    # while all the "small" fields get flushed. Possibly not much to be
                    # done about it.
                    core.update_progress(1)
            logger.info(
                f"Finished reading VCF for partition {partition_index}, "
                f"flushing buffers"
            )

        partition_metadata = IcfPartitionMetadata(
            num_records=num_records,
            last_position=last_position,
            field_summaries=tcw.field_summaries,
        )
        with open(summary_path, "w") as f:
            f.write(partition_metadata.asjson())
        logger.info(
            f"Finish p{partition_index} {partition.vcf_path}__{partition.region} "
            f"{num_records} records last_pos={last_position}"
        )

    def explode(self, *, worker_processes=1, show_progress=False):
        self.load_metadata()
        num_records = self.metadata.num_records
        if np.isinf(num_records):
            logger.warning(
                "Total records unknown, cannot show progress; "
                "reindex VCFs with bcftools index to fix"
            )
            num_records = None
        num_fields = len(self.metadata.fields)
        num_samples = len(self.metadata.samples)
        logger.info(
            f"Exploding fields={num_fields} samples={num_samples}; "
            f"partitions={self.num_partitions} "
            f"variants={'unknown' if num_records is None else num_records}"
        )
        progress_config = core.ProgressConfig(
            total=num_records,
            units="vars",
            title="Explode",
            show=show_progress,
        )
        with core.ParallelWorkManager(worker_processes, progress_config) as pwm:
            for j in range(self.num_partitions):
                pwm.submit(self.process_partition, j)

    def explode_partition(self, partition):
        self.load_metadata()
        if partition < 0 or partition >= self.num_partitions:
            raise ValueError("Partition index not in the valid range")
        self.process_partition(partition)

    def finalise(self):
        self.load_metadata()
        partition_summaries = self.load_partition_summaries()
        total_records = 0
        for index, summary in enumerate(partition_summaries):
            partition_records = summary.num_records
            self.metadata.partitions[index].num_records = partition_records
            self.metadata.partitions[index].region.end = summary.last_position
            total_records += partition_records
        if not np.isinf(self.metadata.num_records):
            # Note: this is just telling us that there's a bug in the
            # index based record counting code, but it doesn't actually
            # matter much. We may want to just make this a warning if
            # we hit regular problems.
            assert total_records == self.metadata.num_records
        self.metadata.num_records = total_records

        check_overlapping_partitions(self.metadata.partitions)

        for field in self.metadata.fields:
            for summary in partition_summaries:
                field.summary.update(summary.field_summaries[field.full_name])

        logger.info("Finalising metadata")
        with open(self.path / "metadata.json", "w") as f:
            f.write(self.metadata.asjson())

        logger.debug("Removing WIP directory")
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
    writer.init(
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


def explode_partition(icf_path, partition):
    writer = IntermediateColumnarFormatWriter(icf_path)
    writer.explode_partition(partition)


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
        # TODO WARNING IF this is a string
        chunk_items = self.chunks[0]
        for size in self.shape[1:]:
            chunk_items *= size
        dt = np.dtype(self.dtype)
        return chunk_items * dt.itemsize


ZARR_SCHEMA_FORMAT_VERSION = "0.3"


@dataclasses.dataclass
class VcfZarrSchema:
    format_version: str
    samples_chunk_size: int
    variants_chunk_size: int
    dimensions: list
    samples: list
    contigs: list
    filters: list
    fields: dict

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
        ret.samples = [Sample(**sd) for sd in d["samples"]]
        ret.contigs = [Contig(**sd) for sd in d["contigs"]]
        ret.filters = [Filter(**sd) for sd in d["filters"]]
        ret.fields = {
            key: ZarrColumnSpec(**value) for key, value in d["fields"].items()
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
            fields={col.name: col for col in colspecs},
            dimensions=["variants", "samples", "ploidy", "alleles", "filters"],
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


def parse_max_memory(max_memory):
    if max_memory is None:
        # Effectively unbounded
        return 2**63
    if isinstance(max_memory, str):
        max_memory = humanfriendly.parse_size(max_memory)
    logger.info(f"Set memory budget to {display_size(max_memory)}")
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
class VcfZarrWriterMetadata:
    format_version: str
    icf_path: str
    schema: VcfZarrSchema
    dimension_separator: str
    partitions: list
    provenance: dict

    def asdict(self):
        return dataclasses.asdict(self)

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
class VcfZarrWriteSummary:
    num_partitions: int
    num_samples: int
    num_variants: int
    num_chunks: int
    max_encoding_memory: str

    def asdict(self):
        return dataclasses.asdict(self)

    def asjson(self):
        return json.dumps(self.asdict(), indent=4)


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
        for field in self.schema.fields.values():
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
            max_encoding_memory=display_size(self.get_max_encoding_memory()),
        )

    def encode_samples(self, root):
        if self.schema.samples != self.icf.metadata.samples:
            raise ValueError(
                "Subsetting or reordering samples not supported currently"
            )  # NEEDS TEST
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
            self.icf = IntermediateColumnarFormat(self.metadata.icf_path)

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
        for col in self.schema.fields.values():
            if col.vcf_field is not None:
                self.encode_array_partition(col, partition_index)
        if "call_genotype" in self.schema.fields:
            self.encode_genotypes_partition(partition_index)

        final_path = self.partition_path(partition_index)
        logger.info(f"Finalising {partition_index} at {final_path}")
        if final_path.exists():
            logger.warning(f"Removing existing partition at {final_path}")
            shutil.rmtree(final_path)
        os.rename(partition_path, final_path)

    def init_partition_array(self, partition_index, name):
        wip_path = self.wip_partition_array_path(partition_index, name)
        # Create an empty array like the definition
        src = self.arrays_path / name
        # Overwrite any existing WIP files
        shutil.copytree(src, wip_path, dirs_exist_ok=True)
        array = zarr.open(wip_path)
        logger.debug(f"Opened empty array {array} @ {wip_path}")
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
            alleles.buff[j, :] = STR_FILL
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
                vid.buff[j] = STR_MISSING
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

        logger.info("Scanning {self.num_partitions} partitions")
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
            for name in self.schema.fields:
                pwm.submit(self.finalise_array, name)
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
        # NOTE This size number is also not quite enough, you need a bit of
        # headroom with it (probably 10% or so). We should include this.
        # FIXME this is actively wrong for String columns. See if we can do better.
        max_encoding_mem = max(
            col.variant_chunk_nbytes for col in self.schema.fields.values()
        )
        gt_mem = 0
        if "call_genotype" in self.schema.fields:
            encoded_together = [
                "call_genotype",
                "call_genotype_phased",
                "call_genotype_mask",
            ]
            gt_mem = sum(
                self.schema.fields[col].variant_chunk_nbytes for col in encoded_together
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
            f"{worker_processes} workers and {display_size(per_worker_memory)} "
            "per worker"
        )
        # Each partition requires per_worker_memory bytes, so to prevent more that
        # max_memory being used, we clamp the number of workers
        max_num_workers = max_memory // per_worker_memory
        if max_num_workers < worker_processes:
            logger.warning(
                f"Limiting number of workers to {max_num_workers} to "
                f"keep within specified memory budget of {display_size(max_memory)}"
            )
        if max_num_workers <= 0:
            raise ValueError(
                f"Insufficient memory to encode a partition:"
                f"{display_size(per_worker_memory)} > {display_size(max_memory)}"
            )
        num_workers = min(max_num_workers, worker_processes)

        total_bytes = 0
        for col in self.schema.fields.values():
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
    icf = IntermediateColumnarFormat(if_path)
    spec = VcfZarrSchema.generate(icf)
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
    icf = IntermediateColumnarFormat(icf_path)
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
        with open(schema_path) as f:
            schema = VcfZarrSchema.fromjson(f.read())
    zarr_path = pathlib.Path(zarr_path)
    vzw = VcfZarrWriter(zarr_path)
    return vzw.init(
        icf,
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
        assert False  # noqa PT015


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
        assert False  # noqa PT015


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
        iterator = tqdm.tqdm(vcf, desc="  Verify", total=vcf.num_records)  # NEEDS TEST
    else:
        iterator = vcf
    for j, row in enumerate(iterator, start_index):
        assert chrom[j] == row.CHROM
        assert pos[j] == row.POS
        assert vid[j] == ("." if row.ID is None else row.ID)
        assert allele[j, 0] == row.REF
        k = len(row.ALT)
        nt.assert_array_equal(allele[j, 1 : k + 1], row.ALT)
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
