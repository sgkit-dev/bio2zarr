import collections
import contextlib
import dataclasses
import json
import logging
import math
import pathlib
import pickle
import re
import shutil
import sys
import tempfile
from functools import partial
from typing import Any

import numcodecs
import numpy as np

from . import constants, core, provenance, vcf_utils, vcz

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class VcfFieldSummary(core.JsonDataclass):
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

    @staticmethod
    def fromdict(d):
        return VcfFieldSummary(**d)


@dataclasses.dataclass(order=True)
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

    @property
    def max_number(self):
        if self.vcf_number in ("R", "A", "G", "."):
            return self.summary.max_number
        else:
            # use declared number if larger than max found
            return max(self.summary.max_number, int(self.vcf_number))

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


ICF_METADATA_FORMAT_VERSION = "0.4"
ICF_DEFAULT_COMPRESSOR = numcodecs.Blosc(
    cname="zstd", clevel=7, shuffle=numcodecs.Blosc.NOSHUFFLE
)


@dataclasses.dataclass
class IcfMetadata(core.JsonDataclass):
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
        d["samples"] = [vcz.Sample(**sd) for sd in d["samples"]]
        d["filters"] = [vcz.Filter(**fd) for fd in d["filters"]]
        d["contigs"] = [vcz.Contig(**cd) for cd in d["contigs"]]
        return IcfMetadata(**d)

    def __eq__(self, other):
        if not isinstance(other, IcfMetadata):
            return NotImplemented
        return (
            self.samples == other.samples
            and self.contigs == other.contigs
            and self.filters == other.filters
            and sorted(self.fields) == sorted(other.fields)
        )


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
        make_field_def("rlen", "Integer", "1"),  # computed field
    ]
    return fields


def scan_vcf(path, target_num_partitions):
    with vcf_utils.VcfFile(path) as vcf_file:
        vcf = vcf_file.vcf
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
                filters.append(vcz.Filter(h["ID"], description))

        # Ensure PASS is the first filter if present
        if pass_index > 0:
            pass_filter = filters.pop(pass_index)
            filters.insert(0, pass_filter)

        fields = fixed_vcf_field_definitions()
        for h in vcf.header_iter():
            if h["HeaderType"] in ["INFO", "FORMAT"]:
                field = VcfField.from_header(h)
                if h["HeaderType"] == "FORMAT" and field.name == "GT":
                    field.vcf_type = "Integer"
                    field.vcf_number = "."
                fields.append(field)

        try:
            contig_lengths = vcf.seqlens
        except AttributeError:
            contig_lengths = [None for _ in vcf.seqnames]

        metadata = IcfMetadata(
            samples=[vcz.Sample(sample_id) for sample_id in vcf.samples],
            contigs=[
                vcz.Contig(contig_id, length)
                for contig_id, length in zip(vcf.seqnames, contig_lengths)
            ],
            filters=filters,
            fields=fields,
            partitions=[],
            num_records=sum(vcf_file.contig_record_counts().values()),
        )

        regions = vcf_file.partition_into_regions(num_parts=target_num_partitions)
        for region in regions:
            metadata.partitions.append(
                VcfPartition(
                    # TODO should this be fully resolving the path? Otherwise it's all
                    # relative to the original WD
                    vcf_path=str(path),
                    region=region,
                )
            )
        logger.info(
            f"Split {path} into {len(metadata.partitions)} "
            f"partitions target={target_num_partitions})"
        )
        core.update_progress(1)
        return metadata, vcf.raw_header


def scan_vcfs(
    paths,
    show_progress,
    target_num_partitions,
    worker_processes=core.DEFAULT_WORKER_PROCESSES,
):
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
            pwm.submit(
                scan_vcf,
                path,
                max(1, target_num_partitions // len(paths)),
            )
        results = list(pwm.results_as_completed())

    # Sort to make the ordering deterministic
    results.sort(key=lambda t: t[0].partitions[0].vcf_path)
    # We just take the first header, assuming the others
    # are compatible.
    all_partitions = []
    total_records = 0
    contigs = {}
    for metadata, _ in results:
        for partition in metadata.partitions:
            logger.debug(f"Scanned partition {partition}")
            all_partitions.append(partition)
        for contig in metadata.contigs:
            if contig.id in contigs:
                if contig != contigs[contig.id]:
                    raise ValueError(
                        "Incompatible contig definitions: "
                        f"{contig} != {contigs[contig.id]}"
                    )
            else:
                contigs[contig.id] = contig
        total_records += metadata.num_records
        metadata.num_records = 0
        metadata.partitions = []

    contig_union = list(contigs.values())
    for metadata, _ in results:
        metadata.contigs = contig_union

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


def sanitise_value_bool(shape, value):
    x = True
    if value is None:
        x = False
    return x


def sanitise_value_float_scalar(shape, value):
    x = value
    if value is None:
        x = [constants.FLOAT32_MISSING]
    return x[0]


def sanitise_value_int_scalar(shape, value):
    x = value
    if value is None:
        x = [constants.INT_MISSING]
    else:
        x = sanitise_int_array(value, ndmin=1, dtype=np.int32)
    return x[0]


def sanitise_value_string_scalar(shape, value):
    if value is None:
        return "."
    else:
        return value[0]


def sanitise_value_string_1d(shape, value):
    if value is None:
        return np.full(shape, ".", dtype="O")
    else:
        value = drop_empty_second_dim(value)
        result = np.full(shape, "", dtype=value.dtype)
        result[: value.shape[0]] = value
        return result


def sanitise_value_string_2d(shape, value):
    if value is None:
        return np.full(shape, ".", dtype="O")
    else:
        result = np.full(shape, "", dtype="O")
        if value.ndim == 2:
            result[: value.shape[0], : value.shape[1]] = value
        else:
            # Convert 1D array into 2D with appropriate shape
            for k, val in enumerate(value):
                result[k, : len(val)] = val
        return result


def drop_empty_second_dim(value):
    assert len(value.shape) == 1 or value.shape[1] == 1
    if len(value.shape) == 2 and value.shape[1] == 1:
        value = value[..., 0]
    return value


def sanitise_value_float_1d(shape, value):
    if value is None:
        return np.full(shape, constants.FLOAT32_MISSING)
    else:
        value = np.array(value, ndmin=1, dtype=np.float32, copy=True)
        # numpy will map None values to Nan, but we need a
        # specific NaN
        value[np.isnan(value)] = constants.FLOAT32_MISSING
        value = drop_empty_second_dim(value)
        result = np.full(shape, constants.FLOAT32_FILL, dtype=np.float32)
        result[: value.shape[0]] = value
        return result


def sanitise_value_float_2d(shape, value):
    if value is None:
        return np.full(shape, constants.FLOAT32_MISSING)
    else:
        value = np.array(value, ndmin=2, dtype=np.float32, copy=True)
        result = np.full(shape, constants.FLOAT32_FILL, dtype=np.float32)
        result[:, : value.shape[1]] = value
        return result


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


def sanitise_value_int_1d(shape, value):
    if value is None:
        return np.full(shape, -1)
    else:
        value = sanitise_int_array(value, 1, np.int32)
        value = drop_empty_second_dim(value)
        result = np.full(shape, -2, dtype=np.int32)
        result[: value.shape[0]] = value
        return result


def sanitise_value_int_2d(shape, value):
    if value is None:
        return np.full(shape, -1)
    else:
        value = sanitise_int_array(value, 2, np.int32)
        result = np.full(shape, -2, dtype=np.int32)
        result[:, : value.shape[1]] = value
        return result


missing_value_map = {
    "Integer": constants.INT_MISSING,
    "Float": constants.FLOAT32_MISSING,
    "String": constants.STR_MISSING,
    "Character": constants.STR_MISSING,
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
        value = np.array(vcf_value, ndmin=self.dimension, copy=True)
        return value

    def transform_and_update_bounds(self, vcf_value):
        if vcf_value is None:
            return None
        # print(self, self.field.full_name, "T", vcf_value)
        value = self.transform(vcf_value)
        self.update_bounds(value)
        return value


class IntegerValueTransformer(VcfValueTransformer):
    def update_bounds(self, value):
        summary = self.field.summary
        # Mask out missing and fill values
        # print(value)
        a = value[value >= constants.MIN_INT_VALUE]
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
        if self.field.category == "FORMAT":
            number = max(len(v) for v in value)
        else:
            number = value.shape[-1]
        # TODO would be nice to report string lengths, but not
        # really necessary.
        summary.max_number = max(summary.max_number, number)

    def transform(self, vcf_value):
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
            chunk_num_records[start_chunk:],
            chunk_cumulative_records[start_chunk + 1 :],
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
        assert len(shape) <= 2
        if self.vcf_field.vcf_type == "Flag":
            assert len(shape) == 0
            return partial(sanitise_value_bool, shape)
        elif self.vcf_field.vcf_type == "Float":
            if len(shape) == 0:
                return partial(sanitise_value_float_scalar, shape)
            elif len(shape) == 1:
                return partial(sanitise_value_float_1d, shape)
            else:
                return partial(sanitise_value_float_2d, shape)
        elif self.vcf_field.vcf_type == "Integer":
            if len(shape) == 0:
                return partial(sanitise_value_int_scalar, shape)
            elif len(shape) == 1:
                return partial(sanitise_value_int_1d, shape)
            else:
                return partial(sanitise_value_int_2d, shape)
        else:
            assert self.vcf_field.vcf_type in ("String", "Character")
            if len(shape) == 0:
                return partial(sanitise_value_string_scalar, shape)
            elif len(shape) == 1:
                return partial(sanitise_value_string_1d, shape)
            else:
                return partial(sanitise_value_string_2d, shape)


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


def convert_local_allele_field_types(fields, schema_instance):
    """
    Update the specified list of fields to include the LAA field, and to convert
    any supported localisable fields to the L* counterpart.

    Note that we currently support only two ALT alleles per sample, and so the
    dimensions of these fields are fixed by that requirement. Later versions may
    use summary data storted in the ICF to make different choices, if information
    about subsequent alleles (not in the actual genotype calls) should also be
    stored.
    """
    fields_by_name = {field.name: field for field in fields}
    gt = fields_by_name["call_genotype"]

    if schema_instance.get_shape(["ploidy"])[0] != 2:
        raise ValueError("Local alleles only supported on diploid data")

    dimensions = gt.dimensions[:-1]

    la = vcz.ZarrArraySpec(
        name="call_LA",
        dtype="i1",
        dimensions=(*dimensions, "local_alleles"),
        description=(
            "0-based indices into REF+ALT, indicating which alleles"
            " are relevant (local) for the current sample"
        ),
    )
    schema_instance.dimensions["local_alleles"] = vcz.VcfZarrDimension.unchunked(
        schema_instance.dimensions["ploidy"].size
    )

    ad = fields_by_name.get("call_AD", None)
    if ad is not None:
        # TODO check if call_LAD is in the list already
        ad.name = "call_LAD"
        ad.source = None
        ad.dimensions = (*dimensions, "local_alleles_AD")
        ad.description += " (local-alleles)"
        schema_instance.dimensions["local_alleles_AD"] = vcz.VcfZarrDimension.unchunked(
            2
        )

    pl = fields_by_name.get("call_PL", None)
    if pl is not None:
        # TODO check if call_LPL is in the list already
        pl.name = "call_LPL"
        pl.source = None
        pl.description += " (local-alleles)"
        pl.dimensions = (*dimensions, "local_" + pl.dimensions[-1].split("_")[-1])
        schema_instance.dimensions["local_" + pl.dimensions[-1].split("_")[-1]] = (
            vcz.VcfZarrDimension.unchunked(3)
        )

    return [*fields, la]


class IntermediateColumnarFormat(vcz.Source):
    def __init__(self, path):
        self._path = pathlib.Path(path)
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
        self.gt_field = None
        for field in self.metadata.fields:
            self.fields[field.full_name] = IntermediateColumnarFormatField(self, field)
            if field.name == "GT":
                self.gt_field = field

        logger.info(
            f"Loaded IntermediateColumnarFormat(partitions={self.num_partitions}, "
            f"records={self.num_records}, fields={self.num_fields})"
        )

    def __repr__(self):
        return (
            f"IntermediateColumnarFormat(fields={len(self.fields)}, "
            f"partitions={self.num_partitions}, "
            f"records={self.num_records}, path={self.path})"
        )

    def summary_table(self):
        data = []
        for name, icf_field in self.fields.items():
            summary = icf_field.vcf_field.summary
            d = {
                "name": name,
                "type": icf_field.vcf_field.vcf_type,
                "chunks": summary.num_chunks,
                "size": core.display_size(summary.uncompressed_size),
                "compressed": core.display_size(summary.compressed_size),
                "max_n": summary.max_number,
                "min_val": core.display_number(summary.min_value),
                "max_val": core.display_number(summary.max_value),
            }

            data.append(d)
        return data

    @property
    def path(self):
        return self._path

    @property
    def num_records(self):
        return self.metadata.num_records

    @property
    def num_partitions(self):
        return len(self.metadata.partitions)

    @property
    def samples(self):
        return self.metadata.samples

    @property
    def contigs(self):
        return self.metadata.contigs

    @property
    def filters(self):
        return self.metadata.filters

    @property
    def num_samples(self):
        return len(self.metadata.samples)

    @property
    def num_fields(self):
        return len(self.fields)

    @property
    def root_attrs(self):
        meta_information_pattern = re.compile("##([^=]+)=(.*)")
        vcf_meta_information = []
        for line in self.vcf_header.split("\n"):
            match = re.fullmatch(meta_information_pattern, line)
            if match:
                key = match.group(1)
                if key in ("contig", "FILTER", "INFO", "FORMAT"):
                    # these fields are stored in Zarr arrays
                    continue
                value = match.group(2)
                vcf_meta_information.append((key, value))
        return {
            "vcf_meta_information": vcf_meta_information,
        }

    def iter_id(self, start, stop):
        for value in self.fields["ID"].iter_values(start, stop):
            if value is not None:
                yield value[0]
            else:
                yield None

    def iter_filters(self, start, stop):
        source_field = self.fields["FILTERS"]
        lookup = {filt.id: index for index, filt in enumerate(self.metadata.filters)}

        for filter_values in source_field.iter_values(start, stop):
            filters = np.zeros(len(self.metadata.filters), dtype=bool)
            if filter_values is not None:
                for filter_id in filter_values:
                    try:
                        filters[lookup[filter_id]] = True
                    except KeyError:
                        raise ValueError(
                            f"Filter '{filter_id}' was not defined in the header."
                        ) from None
            yield filters

    def iter_contig(self, start, stop):
        source_field = self.fields["CHROM"]
        lookup = {
            contig.id: index for index, contig in enumerate(self.metadata.contigs)
        }

        for value in source_field.iter_values(start, stop):
            # Note: because we are using the indexes to define the lookups
            # and we always have an index, it seems that we the contig lookup
            # will always succeed. However, if anyone ever does hit a KeyError
            # here, please do open an issue with a reproducible example!
            yield lookup[value[0]]

    def iter_field(self, field_name, shape, start, stop):
        source_field = self.fields[field_name]
        sanitiser = source_field.sanitiser_factory(shape)
        for value in source_field.iter_values(start, stop):
            yield sanitiser(value)

    def iter_alleles(self, start, stop, num_alleles):
        ref_field = self.fields["REF"]
        alt_field = self.fields["ALT"]

        for ref, alt in zip(
            ref_field.iter_values(start, stop),
            alt_field.iter_values(start, stop),
        ):
            alleles = np.full(num_alleles, constants.STR_FILL, dtype="O")
            alleles[0] = ref[0]
            alleles[1 : 1 + len(alt)] = alt
            yield alleles

    def iter_genotypes(self, shape, start, stop):
        source_field = self.fields["FORMAT/GT"]
        for value in source_field.iter_values(start, stop):
            genotypes = value[:, :-1] if value is not None else None
            phased = value[:, -1] if value is not None else None
            sanitised_genotypes = sanitise_value_int_2d(shape, genotypes)
            sanitised_phased = sanitise_value_int_1d(shape[:-1], phased)
            # Force haploids to always be phased
            # https://github.com/sgkit-dev/bio2zarr/issues/399
            if sanitised_genotypes.shape[1] == 1:
                sanitised_phased[:] = True
            yield sanitised_genotypes, sanitised_phased

    def iter_alleles_and_genotypes(self, start, stop, shape, num_alleles):
        variant_lengths = self.fields["rlen"].iter_values(start, stop)
        if self.gt_field is None or shape is None:
            for variant_length, alleles in zip(
                variant_lengths, self.iter_alleles(start, stop, num_alleles)
            ):
                yield vcz.VariantData(variant_length, alleles, None, None)
        else:
            for variant_length, alleles, (gt, phased) in zip(
                variant_lengths,
                self.iter_alleles(start, stop, num_alleles),
                self.iter_genotypes(shape, start, stop),
            ):
                yield vcz.VariantData(variant_length, alleles, gt, phased)

    def generate_schema(
        self, variants_chunk_size=None, samples_chunk_size=None, local_alleles=None
    ):
        if local_alleles is None:
            local_alleles = False

        max_alleles = max(self.fields["ALT"].vcf_field.summary.max_number + 1, 2)

        # Add ploidy and genotypes dimensions only when needed
        max_genotypes = 0
        for field in self.metadata.format_fields:
            if field.vcf_number == "G":
                max_genotypes = max(max_genotypes, field.summary.max_number)

        ploidy = None
        genotypes_size = None
        if self.gt_field is not None:
            ploidy = max(self.gt_field.summary.max_number - 1, 1)
            # NOTE: it's not clear why we're computing this, when we must have had
            # at least one number=G field to require it anyway?
            genotypes_size = math.comb(max_alleles + ploidy - 1, ploidy)
            # assert max_genotypes == genotypes_size
        else:
            if max_genotypes > 0:
                # there is no GT field, but there is at least one Number=G field,
                # so need to define genotypes dimension
                genotypes_size = max_genotypes

        dimensions = vcz.standard_dimensions(
            variants_size=self.num_records,
            variants_chunk_size=variants_chunk_size,
            samples_size=self.num_samples,
            samples_chunk_size=samples_chunk_size,
            alleles_size=max_alleles,
            filters_size=self.metadata.num_filters,
            ploidy_size=ploidy,
            genotypes_size=genotypes_size,
        )

        schema_instance = vcz.VcfZarrSchema(
            format_version=vcz.ZARR_SCHEMA_FORMAT_VERSION,
            dimensions=dimensions,
            fields=[],
        )

        logger.info(
            "Generating schema with chunks="
            f"variants={dimensions['variants'].chunk_size}, "
            f"samples={dimensions['samples'].chunk_size}"
        )

        def spec_from_field(field, array_name=None):
            return vcz.ZarrArraySpec.from_field(
                field,
                schema_instance,
                array_name=array_name,
            )

        def fixed_field_spec(name, dtype, source=None, dimensions=("variants",)):
            compressor = (
                vcz.DEFAULT_ZARR_COMPRESSOR_BOOL.get_config()
                if dtype == "bool"
                else None
            )
            return vcz.ZarrArraySpec(
                source=source,
                name=name,
                dtype=dtype,
                description="",
                dimensions=dimensions,
                compressor=compressor,
            )

        name_map = {field.full_name: field for field in self.metadata.fields}
        array_specs = [
            fixed_field_spec(
                name="variant_contig",
                dtype=core.min_int_dtype(0, self.metadata.num_contigs),
            ),
            fixed_field_spec(
                name="variant_filter",
                dtype="bool",
                dimensions=["variants", "filters"],
            ),
            fixed_field_spec(
                name="variant_allele",
                dtype="O",
                dimensions=["variants", "alleles"],
            ),
            fixed_field_spec(
                name="variant_length",
                dtype=name_map["rlen"].smallest_dtype(),
                dimensions=["variants"],
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

        # Only two of the fixed fields have a direct one-to-one mapping.
        array_specs.extend(
            [
                spec_from_field(name_map["QUAL"], array_name="variant_quality"),
                spec_from_field(name_map["POS"], array_name="variant_position"),
            ]
        )
        array_specs.extend(
            [spec_from_field(field) for field in self.metadata.info_fields]
        )

        for field in self.metadata.format_fields:
            if field.name == "GT":
                continue
            array_specs.append(spec_from_field(field))

        if self.gt_field is not None and self.num_samples > 0:
            array_specs.append(
                vcz.ZarrArraySpec(
                    name="call_genotype_phased",
                    dtype="bool",
                    dimensions=["variants", "samples"],
                    description="",
                    compressor=vcz.DEFAULT_ZARR_COMPRESSOR_BOOL.get_config(),
                )
            )
            array_specs.append(
                vcz.ZarrArraySpec(
                    name="call_genotype",
                    dtype=self.gt_field.smallest_dtype(),
                    dimensions=["variants", "samples", "ploidy"],
                    description="",
                    compressor=vcz.DEFAULT_ZARR_COMPRESSOR_GENOTYPES.get_config(),
                )
            )
            array_specs.append(
                vcz.ZarrArraySpec(
                    name="call_genotype_mask",
                    dtype="bool",
                    dimensions=["variants", "samples", "ploidy"],
                    description="",
                    compressor=vcz.DEFAULT_ZARR_COMPRESSOR_BOOL.get_config(),
                )
            )

        if local_alleles:
            array_specs = convert_local_allele_field_types(array_specs, schema_instance)

        schema_instance.fields = array_specs
        return schema_instance


@dataclasses.dataclass
class IcfPartitionMetadata(core.JsonDataclass):
    num_records: int
    last_position: int
    field_summaries: dict

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
class IcfWriteSummary(core.JsonDataclass):
    num_partitions: int
    num_samples: int
    num_variants: int


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
        worker_processes=core.DEFAULT_WORKER_PROCESSES,
        target_num_partitions=None,
        show_progress=False,
        compressor=None,
    ):
        if self.path.exists():
            raise ValueError(f"ICF path already exists: {self.path}")
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
            field_path = get_vcf_field_path(self.path, field)
            field_path.mkdir(parents=True)

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
            with vcf_utils.VcfFile(partition.vcf_path) as vcf:
                num_records = 0
                for variant in vcf.variants(partition.region):
                    num_records += 1
                    last_position = variant.POS
                    tcw.append("CHROM", variant.CHROM)
                    tcw.append("POS", variant.POS)
                    tcw.append("QUAL", variant.QUAL)
                    tcw.append("ID", variant.ID)
                    tcw.append("FILTERS", variant.FILTERS)
                    tcw.append("REF", variant.REF)
                    tcw.append("ALT", variant.ALT)
                    tcw.append("rlen", variant.end - variant.start)
                    for field in info_fields:
                        tcw.append(field.full_name, variant.INFO.get(field.name, None))
                    if has_gt:
                        val = None
                        if "GT" in variant.FORMAT and variant.genotype is not None:
                            val = variant.genotype.array()
                        tcw.append("FORMAT/GT", val)
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

    def explode(
        self, *, worker_processes=core.DEFAULT_WORKER_PROCESSES, show_progress=False
    ):
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
    worker_processes=core.DEFAULT_WORKER_PROCESSES,
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
    worker_processes=core.DEFAULT_WORKER_PROCESSES,
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
    if not path.exists():
        raise ValueError(f"Path not found: {path}")
    if (path / "metadata.json").exists():
        obj = IntermediateColumnarFormat(path)
    # NOTE: this is too strict, we should support more general Zarrs, see #276
    elif (path / ".zmetadata").exists():
        obj = vcz.VcfZarr(path)
    else:
        raise ValueError(f"{path} not in ICF or VCF Zarr format")
    return obj.summary_table()


def mkschema(
    if_path,
    out,
    *,
    variants_chunk_size=None,
    samples_chunk_size=None,
    local_alleles=None,
):
    store = IntermediateColumnarFormat(if_path)
    spec = store.generate_schema(
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
        local_alleles=local_alleles,
    )
    out.write(spec.asjson())


def convert(
    vcfs,
    vcz_path,
    *,
    variants_chunk_size=None,
    samples_chunk_size=None,
    worker_processes=core.DEFAULT_WORKER_PROCESSES,
    local_alleles=None,
    show_progress=False,
    icf_path=None,
):
    """
    Convert the VCF data at the specified list of paths
    to VCF Zarr format stored at the specified path.

    .. todo:: Document parameters
    """
    if icf_path is None:
        cm = temp_icf_path(prefix="vcf2zarr")
    else:
        cm = contextlib.nullcontext(icf_path)

    with cm as icf_path:
        explode(
            icf_path,
            vcfs,
            worker_processes=worker_processes,
            show_progress=show_progress,
        )
        encode(
            icf_path,
            vcz_path,
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


def encode(
    icf_path,
    zarr_path,
    schema_path=None,
    variants_chunk_size=None,
    samples_chunk_size=None,
    max_variant_chunks=None,
    dimension_separator=None,
    max_memory=None,
    local_alleles=None,
    worker_processes=core.DEFAULT_WORKER_PROCESSES,
    show_progress=False,
):
    # Rough heuristic to split work up enough to keep utilisation high
    target_num_partitions = max(1, worker_processes * 4)
    encode_init(
        icf_path,
        zarr_path,
        target_num_partitions,
        schema_path=schema_path,
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
        local_alleles=local_alleles,
        max_variant_chunks=max_variant_chunks,
        dimension_separator=dimension_separator,
    )
    vzw = vcz.VcfZarrWriter(IntermediateColumnarFormat, zarr_path)
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
    worker_processes=core.DEFAULT_WORKER_PROCESSES,
    show_progress=False,
):
    icf_store = IntermediateColumnarFormat(icf_path)
    if schema_path is None:
        schema_instance = icf_store.generate_schema(
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
            schema_instance = vcz.VcfZarrSchema.fromjson(f.read())
    zarr_path = pathlib.Path(zarr_path)
    vzw = vcz.VcfZarrWriter("icf", zarr_path)
    return vzw.init(
        icf_store,
        target_num_partitions=target_num_partitions,
        schema=schema_instance,
        dimension_separator=dimension_separator,
        max_variant_chunks=max_variant_chunks,
    )


def encode_partition(zarr_path, partition):
    writer_instance = vcz.VcfZarrWriter(IntermediateColumnarFormat, zarr_path)
    writer_instance.encode_partition(partition)


def encode_finalise(zarr_path, show_progress=False):
    writer_instance = vcz.VcfZarrWriter(IntermediateColumnarFormat, zarr_path)
    writer_instance.finalise(show_progress=show_progress)
