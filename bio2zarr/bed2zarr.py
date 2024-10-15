import dataclasses
import json
import logging
import math
import pathlib
import shutil
from enum import Enum
from typing import Any

import numcodecs
import numpy as np
import pandas as pd
import zarr

from . import core, provenance

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


class BedColumns(Enum):
    contig = 0
    start = 1
    end = 2
    name = 3
    score = 4
    strand = 5
    thickStart = 6
    thickEnd = 7
    itemRgb = 8
    blockCount = 9
    blockSizes = 10
    blockStarts = 11


@dataclasses.dataclass
class ZarrArraySpec:
    name: str
    dtype: str
    shape: list
    chunks: list
    dimensions: tuple
    description: str
    compressor: dict
    filters: list

    def __post_init__(self):
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

    def _choose_compressor_settings(self):
        pass

    @staticmethod
    def from_field(
        bed_field,
        *,
        num_records,
        records_chunk_size,
    ):
        shape = [num_records]
        dimensions = ["records"]
        chunks = [records_chunk_size]
        return ZarrArraySpec.new(
            name=bed_field.name,
            dtype=bed_field.smallest_dtype(),
            shape=shape,
            chunks=chunks,
            dimensions=dimensions,
            description=bed_field.description,
        )


ZARR_SCHEMA_FORMAT_VERSION = "0.1"


@dataclasses.dataclass
class Contig:
    id: str


@dataclasses.dataclass
class Name:
    id: str


@dataclasses.dataclass
class BedMetadata(core.JsonDataclass):
    contigs: list
    names: list
    fields: list
    bed_type: BedType
    num_records: int = -1
    records_chunk_size: int = 1000

    @property
    def num_contigs(self):
        return len(self.contigs)

    @property
    def num_names(self):
        return len(self.names)


@dataclasses.dataclass
class BedZarrSchema(core.JsonDataclass):
    format_version: str
    fields: list
    contigs: list
    names: list
    bed_type: str
    records_chunk_size: int

    def validate(self):
        """
        Checks that the schema is well-formed and within required limits.
        """

    @staticmethod
    def fromdict(d):
        if d["format_version"] != ZARR_SCHEMA_FORMAT_VERSION:
            raise ValueError(
                "BedZarrSchema format version mismatch: "
                f"{d['format_version']} != {ZARR_SCHEMA_FORMAT_VERSION}"
            )
        ret = BedZarrSchema(**d)
        ret.fields = [ZarrArraySpec(**sd) for sd in d["fields"]]
        ret.contig_ids = [Contig(**sd) for sd in d["contigs"]]
        ret.name_ids = [Name(**sd) for sd in d["names"]]
        return ret

    @staticmethod
    def fromjson(s):
        return BedZarrSchema.fromdict(json.loads(s))

    @staticmethod
    def generate(metadata, records_chunk_size=None):
        if records_chunk_size is None:
            records_chunk_size = 1000
        logger.info("Generating schema with chunks=%d", records_chunk_size)
        fields = metadata.fields

        def spec_from_field(field):
            return ZarrArraySpec.from_field(
                field,
                num_records=metadata.num_records,
                records_chunk_size=records_chunk_size,
            )

        specs = [spec_from_field(f) for f in fields]

        # Contig_id and name_id specs unnecessary?
        contig_id_spec = ZarrArraySpec.new(
            name="contig_id",
            dtype="O",
            shape=[metadata.num_contigs],
            chunks=[metadata.num_contigs],
            dimensions=["contigs"],
            description="Contig ID",
        )
        name_id_spec = ZarrArraySpec.new(
            name="name_id",
            dtype="O",
            shape=[metadata.num_names],
            chunks=[metadata.num_names],
            dimensions=["names"],
            description="Name ID",
        )
        specs.extend([contig_id_spec, name_id_spec])
        return BedZarrSchema(
            format_version=ZARR_SCHEMA_FORMAT_VERSION,
            fields=specs,
            contigs=metadata.contigs,
            names=metadata.names,
            bed_type=metadata.bed_type.name,
            records_chunk_size=records_chunk_size,
        )


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


BZW_METADATA_FORMAT_VERSION = "0.1"


@dataclasses.dataclass
class BedZarrWriterMetadata(core.JsonDataclass):
    format_version: str
    fields: list
    contigs: list
    names: list
    bed_type: str
    schema: BedZarrSchema

    @staticmethod
    def fromdict(d):
        if d["format_version"] != BZW_METADATA_FORMAT_VERSION:
            raise ValueError(
                "BedZarrWriterMetadata format version mismatch: "
                f"{d['format_version']} != {BZW_METADATA_FORMAT_VERSION}"
            )
        ret = BedZarrWriterMetadata(**d)
        ret.schema = BedZarrSchema.fromdict(d["schema"])
        return ret


class BedZarrWriter:
    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.wip_path = self.path / "wip"
        self.data = None
        self.metadata = None
        self.bed_path = None
        self.bed_type = None
        self.fields = None

    @property
    def schema(self):
        if self.metadata is None:
            return
        return self.metadata.schema

    def init(
        self,
        data,
        *,
        metadata,
        bed_path,
        schema,
    ):
        self.data = data
        self.bed_metadata = metadata
        self.bed_path = bed_path

        if self.path.exists():
            raise ValueError("Zarr path already exists")
        schema.validate()

        self.metadata = BedZarrWriterMetadata(
            format_version=BZW_METADATA_FORMAT_VERSION,
            bed_type=self.bed_metadata.bed_type.name,
            fields=self.fields,
            contigs=self.bed_metadata.contigs,
            names=self.bed_metadata.names,
            schema=schema,
        )
        self.path.mkdir()
        self.wip_path.mkdir()
        logger.info("Writing WIP metadata")
        with open(self.wip_path / "metadata.json", "w") as f:
            json.dump(self.metadata.asdict(), f, indent=4)

    def write(self):
        store = zarr.DirectoryStore(self.path)
        root = zarr.group(store=store)
        root.attrs.update(
            {
                "bed_zarr_version": "0.1",
                "source": f"bio2zarr-{provenance.__version__}",
            }
        )
        datafields = self.schema.fields[0 : BedType[self.schema.bed_type].value]
        d = {i: f.name for i, f in enumerate(datafields)}
        self.data.rename(columns=d, inplace=True)
        self.encode_fields(root, datafields)
        self.encode_contig_id(root)
        self.encode_name_id(root)

    def encode_contig_id(self, root):
        array = root.array(
            "contig_id",
            [contig.id for contig in self.schema.contigs],
            dtype="str",
            compressor=DEFAULT_ZARR_COMPRESSOR,
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ["contigs"]

    def encode_name_id(self, root):
        array = root.array(
            "name_id",
            [name.id for name in self.schema.names],
            dtype="str",
            compressor=DEFAULT_ZARR_COMPRESSOR,
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ["names"]

    def finalise(self):
        self.load_metadata()
        logger.debug("Removing %s", self.wip_path)
        shutil.rmtree(self.wip_path)
        logger.info("Consolidating Zarr metadata")
        zarr.consolidate_metadata(self.path)

    def load_metadata(self):
        if self.metadata is None:
            with open(self.wip_path / "metadata.json") as f:
                self.metadata = BedZarrWriterMetadata.fromdict(json.load(f))

    def encode_fields(self, root, datafields):
        for field in datafields:
            object_codec = None
            if field.dtype == "O":
                object_codec = numcodecs.VLenUTF8()
            array = root.array(
                field.name,
                self.data[field.name].values,
                shape=field.shape,
                dtype=field.dtype,
                compressor=DEFAULT_ZARR_COMPRESSOR,
                chunks=(self.schema.records_chunk_size,),
                object_codec=object_codec,
            )
            array.attrs["_ARRAY_DIMENSIONS"] = ["records"]
            logger.debug("%s done", field)


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


def parse_bed(bed_path, records_chunk_size=None):
    """Read and parse BED file and return data frame and metadata."""
    bed_type = guess_bed_file_type(bed_path)
    fields = mkfields(bed_type)
    data = pd.read_table(bed_path, header=None, names=[f.name for f in fields])
    data, contig_id, name_id = encode_categoricals(data, bed_type)
    data = parse_csv_fields(data, bed_type)
    fields = update_field_bounds(bed_type, data)
    metadata = BedMetadata(
        contigs=[Contig(c) for c in contig_id],
        names=[Name(f) for f in name_id],
        fields=fields,
        bed_type=bed_type,
        num_records=data.shape[0],
        records_chunk_size=records_chunk_size,
    )
    return data, metadata


def parse_csv_fields(data, bed_type):
    if bed_type.value < BedType.BED8.value:
        return data

    def _convert_csv_data(values):
        if values.dtype == "O":
            ret = values.str.split(",").apply(lambda x: np.array([int(y) for y in x]))
        else:
            ret = values
        return ret

    if bed_type.value >= BedType.BED9.value:
        data["itemRgb"] = _convert_csv_data(data["itemRgb"])
    if bed_type.value == BedType.BED12.value:
        data["blockSizes"] = _convert_csv_data(data["blockSizes"])
        data["blockStarts"] = _convert_csv_data(data["blockStarts"])
    return data


def encode_categoricals(data, bed_type):
    """Convert categoricals to integer encodings."""
    contig = pd.Categorical(
        data["contig"], categories=data["contig"].unique(), ordered=True
    )
    contig_id = contig.categories.values
    data["contig"] = contig.codes
    if bed_type.value >= BedType.BED4.value:
        name = pd.Categorical(
            data["name"], categories=data["name"].unique(), ordered=True
        )
        name_id = name.categories.values
        data["name"] = name.codes
    else:
        name_id = []
    return data, contig_id, name_id


def update_field_bounds(bed_type, data):
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


def mkschema(bed_path, out):
    """Write schema to file"""
    _, metadata = parse_bed(bed_path)
    spec = BedZarrSchema.generate(
        metadata, records_chunk_size=metadata.records_chunk_size
    )
    out.write(spec.asjson())


def init_schema(schema_path, metadata):
    if schema_path is None:
        schema = BedZarrSchema.generate(
            metadata, records_chunk_size=metadata.records_chunk_size
        )
    else:
        logger.info("Reading schema from %s", schema_path)
        if metadata.records_chunk_size is not None:
            raise ValueError(
                "Cannot specify schema along with chunk sizes"
            )  # NEEDS TEST
        with open(schema_path) as f:
            schema = BedZarrSchema.fromjson(f.read())
    return schema


def bed2zarr(
    bed_path,
    zarr_path,
    schema_path=None,
    records_chunk_size=None,
):
    data, metadata = parse_bed(bed_path, records_chunk_size)
    schema = init_schema(schema_path, metadata)
    bedzw = BedZarrWriter(zarr_path)
    bedzw.init(
        data=data,
        metadata=metadata,
        bed_path=bed_path,
        schema=schema,
    )
    bedzw.write()
    bedzw.finalise()
