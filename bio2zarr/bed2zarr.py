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
class Feature:
    id: str


@dataclasses.dataclass
class BedMetadata(core.JsonDataclass):
    contigs: list
    features: list
    fields: list
    bed_type: BedType
    num_records: int = -1
    records_chunk_size: int = 1000

    @property
    def num_contigs(self):
        return len(self.contigs)

    @property
    def num_features(self):
        return len(self.features)


@dataclasses.dataclass
class BedZarrSchema(core.JsonDataclass):
    format_version: str
    fields: list
    contigs: list
    features: list
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
        ret.contigs = [Contig(**sd) for sd in d["contigs"]]
        ret.features = [Feature(**sd) for sd in d["features"]]
        return ret

    @staticmethod
    def fromjson(s):
        return BedZarrSchema.fromdict(json.loads(s))

    @staticmethod
    def generate(num_records, bed_type, fields, contigs, features, records_chunk_size=None):
        if records_chunk_size is None:
            records_chunk_size = 1000
        logger.info("Generating schema with chunks=%d", records_chunk_size)

        def spec_from_field(field):
            return ZarrArraySpec.from_field(
                field,
                num_records=num_records,
                records_chunk_size=records_chunk_size,
            )

        specs = [spec_from_field(f) for f in fields]

        contig_spec = ZarrArraySpec.new(
            name="contig_id",
            dtype="O",
            shape=[len(contigs)],
            chunks=[len(contigs)],
            dimensions=["contigs"],
            description="Contig ID",
        )
        feature_spec = ZarrArraySpec.new(
            name="feature_id",
            dtype="O",
            shape=[len(features)],
            chunks=[len(features)],
            dimensions=["features"],
            description="Feature ID",
        )
        fields.extend([contig_spec, feature_spec])
        return BedZarrSchema(
            format_version=ZARR_SCHEMA_FORMAT_VERSION,
            fields=specs,
            contigs=contigs,
            features=features,
            bed_type=bed_type.name,
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
        self.num_chunks = array.nchunks
        self.compressed_size = array.nbytes
        self.uncompressed_size = array.nbytes
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
        """Return the smallest dtype suitable for this field based on type and values"""
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
    return BedType(num_cols)


BZW_METADATA_FORMAT_VERSION = "0.1"


@dataclasses.dataclass
class BedZarrWriterMetadata(core.JsonDataclass):
    format_version: str
    fields: list
    contigs: list
    features: list
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
            features=self.bed_metadata.features,
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
        d = {i: f.name for i, f in enumerate(self.schema.fields)}
        self.data.rename(columns=d, inplace=True)
        self.encode_fields(root)
        self.encode_contig_id(root)
        self.encode_feature_id(root)

    def encode_contig_id(self, root):
        print(self.schema.contigs)
        array = root.array(
            "contig_id",
            [contig.id for contig in self.schema.contigs],
            dtype="str",
            compressor=DEFAULT_ZARR_COMPRESSOR,
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ["contigs"]


    def encode_feature_id(self, root):
        array = root.array(
            "feature_id",
            [feature.id for feature in self.schema.features],
            dtype="str",
            compressor=DEFAULT_ZARR_COMPRESSOR,
        )
        array.attrs["_ARRAY_DIMENSIONS"] = ["features"]

        
    def finalise(self):
        logger.debug("Removing %s", self.wip_path)
        shutil.rmtree(self.wip_path)

    def load_metadata(self):
        if self.metadata is None:
            with open(self.wip_path / "metadata.json") as f:
                self.metadata = BedZarrWriterMetadata.fromdict(json.load(f))

    def encode_fields(self, root):
        for field in self.schema.fields:
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
        make_field_def("start", "chromStart", "Integer", "Feature start position"),
        make_field_def("end", "chromEnd", "Integer", "Feature end position"),
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
        make_field_def("name", "name", "Category", "Feature description"),
        make_field_def("score", "score", "Integer", "A numerical value"),
        make_field_def("strand", "strand", "Character", "Feature strand"),
        make_field_def(
            "thickStart", "thickStart", "Integer", "Thick start position"
        ),
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


def read_bed(bed_path, bed_type):
    """Read BED file."""
    fields = mkfields(bed_type)    
    data = pd.read_table(bed_path, header=None,
                         names=[f.name for f in fields])
    return data


def encode_categoricals(data, bed_type):
    """Convert categoricals to integer encodings."""
    contig = pd.Categorical(
        data["contig"], categories=data["contig"].unique(), ordered=True
    )
    contig_id = contig.categories.values
    data["contig"] = contig.codes
    if bed_type.value >= BedType.BED4.value:
        feature = pd.Categorical(
            data["name"], categories=data["name"].unique(), ordered=True
        )
        feature_id = feature.categories.values
        data["name"] = feature.codes
    else:
        feature_id = None
    return data, contig_id, feature_id


def update_field_bounds(bed_type, data):
    """Update field bounds based on data."""
    ret = []
    fields = mkfields(bed_type)
    for f in fields:
        if f.bed_dtype == "Integer":
            if f.name in ("itemRgb", "blockSizes", "blockStarts"):
                if data[f.name].dtype == "O":
                    values = np.concatenate(
                        data[f.name].str.split(",").apply(
                            lambda x: np.array([int(y) for y in x])
                        ))
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


def mkschema(schema_path, metadata):
    """Read schema or make schema from fields."""
    if schema_path is None:
        schema = BedZarrSchema.generate(
            num_records=metadata.num_records,
            bed_type=metadata.bed_type,
            fields=metadata.fields,
            contigs=metadata.contigs,
            features=metadata.features,
            records_chunk_size=metadata.records_chunk_size,
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
    bed_type = guess_bed_file_type(bed_path)
    data = read_bed(bed_path, bed_type)
    data, contig_id, feature_id = encode_categoricals(data, bed_type)
    fields = update_field_bounds(bed_type, data)
    metadata = BedMetadata(
        contigs=[Contig(c) for c in contig_id],
        features=[Feature(f) for f in feature_id],
        fields=fields,
        bed_type=bed_type,
        num_records=data.shape[0],
        records_chunk_size=records_chunk_size,
    )
    schema = mkschema(
        schema_path,
        metadata,
    )
    bedzw = BedZarrWriter(zarr_path)
    bedzw.init(
        data=data,
        metadata=metadata,
        bed_path=bed_path,
        schema=schema,
    )
    bedzw.write()
    bedzw.finalise()
