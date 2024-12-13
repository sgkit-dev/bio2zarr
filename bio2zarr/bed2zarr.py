import dataclasses
import logging
import math
from enum import Enum
from typing import Any

import numcodecs
import numpy as np
import pandas as pd
import xarray as xr

from . import core

logger = logging.getLogger(__name__)

DEFAULT_ZARR_COMPRESSOR = numcodecs.Blosc(cname="zstd", clevel=7)


class BedType(Enum):
    BED3 = 3
    BED4 = 4
    BED5 = 5
    BED6 = 6
    BED7 = 7
    BED8 = 8
    BED9 = 9
    BED12 = 12


@dataclasses.dataclass
class BedFieldSummary(core.JsonDataclass):
    num_chunks: int = 0
    compressed_size: int = 0
    uncompressed_size: int = 0
    max_value: Any = -math.inf
    min_value: Any = math.inf

    def update(self, other):
        self.num_chunks = other.num_chunks
        self.compressed_size = other.compressed_size
        self.uncompressed_size = other.uncompressed_size
        self.min_value = min(self.min_value, other.min_value)
        self.max_value = max(self.max_value, other.max_value)

    @staticmethod
    def fromdict(d):
        return BedFieldSummary(**d)


@dataclasses.dataclass
class BedField:
    category: str
    name: str
    alias: str
    description: str
    bed_dtype: str
    summary: BedFieldSummary

    def smallest_dtype(self):
        """Return the smallest dtype suitable for this field based on
        type and values"""
        s = self.summary
        if self.bed_dtype == "Integer":
            if not math.isfinite(s.max_value):
                ret = "i1"
            else:
                ret = core.min_int_dtype(s.min_value, s.max_value)
        elif self.bed_dtype == "Character":
            ret = "U1"
        elif self.bed_dtype == "Category":
            ret = core.min_int_dtype(s.min_value, s.max_value)
        else:
            assert self.bed_dtype == "String"
            ret = "O"
        return ret


def guess_bed_file_type(path):
    """Check the number of columns in a BED file and guess BED
    type."""
    first = pd.read_table(path, header=None, nrows=1)
    num_cols = first.shape[1]
    if num_cols < 3:
        raise ValueError(f"Expected at least 3 columns in BED file, got {num_cols}")
    if num_cols > 12:
        raise ValueError(f"Expected at most 12 columns in BED file, got {num_cols}")
    if num_cols in (10, 11):
        raise ValueError(f"BED10 and BED11 are prohibited, got {num_cols} columns")
    if num_cols in (9, 12):
        raise ValueError("BED9 and BED12 are valid but currently unsupported formats")
    return BedType(num_cols)


# See https://samtools.github.io/hts-specs/BEDv1.pdf
def mandatory_bed_field_definitions():
    def make_field_def(name, alias, bed_dtype, description=""):
        return BedField(
            category="mandatory",
            name=name,
            alias=alias,
            description=description,
            bed_dtype=bed_dtype,
            summary=BedFieldSummary(),
        )

    fields = [
        make_field_def("contig", "chrom", "Category", "Chromosome name"),
        make_field_def("start", "chromStart", "Integer", "Name start position"),
        make_field_def("end", "chromEnd", "Integer", "Name end position"),
    ]
    return fields


def optional_bed_field_definitions(num_fields=0):
    def make_field_def(name, alias, bed_dtype, description=""):
        return BedField(
            category="optional",
            name=name,
            alias=alias,
            description=description,
            bed_dtype=bed_dtype,
            summary=BedFieldSummary(),
        )

    fields = [
        make_field_def("name", "name", "Category", "Name description"),
        make_field_def("score", "score", "Integer", "A numerical value"),
        make_field_def("strand", "strand", "Character", "Name strand"),
        make_field_def("thickStart", "thickStart", "Integer", "Thick start position"),
        make_field_def("thickEnd", "thickEnd", "Integer", "Thick end position"),
        make_field_def("itemRgb", "itemRgb", "Integer", "Display"),
        make_field_def("blockCount", "blockCount", "Integer", "Number of blocks"),
        make_field_def("blockSizes", "blockSizes", "Integer", "Block sizes"),
        make_field_def(
            "blockStarts", "blockStarts", "Integer", "Block start positions"
        ),
    ]

    return fields[:num_fields]


def mkfields(bed_type):
    mandatory = mandatory_bed_field_definitions()
    optional = optional_bed_field_definitions(bed_type.value - BedType.BED3.value)
    return mandatory + optional


def encode_categoricals(data, bed_type):
    """Convert categoricals to integer encodings."""
    contig = pd.Categorical(
        data["contig"], categories=data["contig"].unique(), ordered=True
    )
    contig_id = contig.categories.values
    contig_id = contig_id.astype(f"<U{len(max(contig_id, key=len))}")
    data["contig"] = contig.codes
    if bed_type.value >= BedType.BED4.value:
        name = pd.Categorical(
            data["name"], categories=data["name"].unique(), ordered=True
        )
        name_id = name.categories.values
        name_id = name_id.astype(f"<U{len(max(name_id, key=len))}")
        data["name"] = name.codes
    else:
        name_id = None
    return data, contig_id, name_id


def update_field_bounds(data, bed_type):
    """Update field bounds based on data."""
    ret = []
    fields = mkfields(bed_type)
    for f in fields:
        if f.bed_dtype == "Integer":
            if f.name in ("itemRgb", "blockSizes", "blockStarts"):
                if data[f.name].dtype == "O":
                    values = np.concatenate(data[f.name])
                else:
                    values = data[f.name]
                f.summary.min_value = min(values)
                f.summary.max_value = max(values)
            else:
                f.summary.min_value = data[f.name].min()
                f.summary.max_value = data[f.name].max()
        elif f.bed_dtype == "Category":
            f.summary.min_value = 0
            f.summary.max_value = max(data[f.name].unique())
        ret.append(f)
    return ret


def bed2zarr(
    bed_path,
    zarr_path,
    records_chunk_size=None,
):
    bed_type = guess_bed_file_type(bed_path)
    fields = mkfields(bed_type)
    data = pd.read_table(bed_path, header=None, names=[f.name for f in fields])
    data, contig_id, name_id = encode_categoricals(data, bed_type)
    fields = update_field_bounds(data, bed_type)
    dtypes = {f.name: f.smallest_dtype() for f in fields}
    data.index.name = "records"
    ds = xr.Dataset.from_dataframe(data)
    for k, v in dtypes.items():
        ds[k] = ds[k].astype(v)
    if records_chunk_size is None:
        records_chunk_size = len(data)
    chunks = {
        "records": records_chunk_size,
        "contigs": len(contig_id),
    }
    ds["contig_id"] = xr.DataArray(contig_id, dims=["contigs"])
    if bed_type.value >= BedType.BED4.value:
        ds["name_id"] = xr.DataArray(name_id, dims=["names"])
        chunks["names"] = len(name_id)
    ds = ds.chunk(chunks)
    ds.to_zarr(zarr_path, mode="w")
