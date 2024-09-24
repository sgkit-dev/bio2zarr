import dataclasses
import json
import logging
import pathlib
import shutil
from enum import Enum

import numcodecs
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
            dtype=bed_field.dtype,
            shape=shape,
            chunks=chunks,
            dimensions=dimensions,
            description=bed_field.description,
        )


ZARR_SCHEMA_FORMAT_VERSION = "0.1"


@dataclasses.dataclass
class BedZarrSchema(core.JsonDataclass):
    format_version: str
    fields: list
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
        return ret

    @staticmethod
    def fromjson(s):
        return BedZarrSchema.fromdict(json.loads(s))

    @staticmethod
    def generate(num_records, bed_type, records_chunk_size=None):
        if records_chunk_size is None:
            records_chunk_size = 1000
        logger.info("Generating schema with chunks=%d", records_chunk_size)

        def spec_from_field(field):
            return ZarrArraySpec.from_field(
                field,
                num_records=num_records,
                records_chunk_size=records_chunk_size,
            )

        fields = mkfields(bed_type)
        specs = [spec_from_field(field) for field in fields]
        return BedZarrSchema(
            format_version=ZARR_SCHEMA_FORMAT_VERSION,
            fields=specs,
            bed_type=bed_type.name,
            records_chunk_size=records_chunk_size,
        )


@dataclasses.dataclass
class BedField:
    category: str
    name: str
    alias: str
    description: str
    dtype: str


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
        self.metadata = None
        self.data = None
        self.bed_path = None
        self.bed_type = None

    @property
    def schema(self):
        return self.metadata.schema

    def init(
        self,
        data,
        *,
        bed_path,
        bed_type,
        schema,
    ):
        self.data = data
        self.bed_type = bed_type
        self.bed_path = bed_path

        if self.path.exists():
            raise ValueError("Zarr path already exists")
        schema.validate()

        fields = mkfields(self.bed_type)
        self.metadata = BedZarrWriterMetadata(
            format_version=BZW_METADATA_FORMAT_VERSION,
            bed_type=self.bed_type.name,
            fields=fields,
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

        # 1. Create contig name <-> contig id mapping

        # 2. update self.data.contig based on mapping (-> also means
        # schema needs modification since we change from string to
        # int!)

        self.encode_fields(root)

    def finalise(self):
        logger.debug("Removing %s", self.wip_path)
        shutil.rmtree(self.wip_path)

    def load_metadata(self):
        if self.metadata is None:
            with open(self.wip_path / "metadata.json") as f:
                self.metadata = BedZarrWriterMetadata.fromdict(json.load(f))

    # FIXME: fields 9-12 are multi-dimensional
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
    def make_field_def(name, alias, dtype, description=""):
        return BedField(
            category="mandatory",
            name=name,
            alias=alias,
            description=description,
            dtype=dtype,
        )

    fields = [
        make_field_def("contig", "chrom", "O", "Chromosome name"),
        make_field_def("start", "chromStart", "i8", "Feature start position"),
        make_field_def("end", "chromEnd", "i8", "Feature end position"),
    ]
    return fields


def optional_bed_field_definitions(num_fields=0):
    def make_field_def(name, alias, dtype=None, *, description=""):
        return BedField(
            category="optional",
            name=name,
            alias=alias,
            description=description,
            dtype=dtype,
        )

    fields = [
        make_field_def("name", "name", "O", description="Feature description"),
        make_field_def("score", "score", "i2", description="A numerical value"),
        make_field_def("strand", "strand", "O", description="Feature strand"),
        make_field_def(
            "thickStart", "thickStart", "i8", description="Thick start position"
        ),
        make_field_def("thickEnd", "thickEnd", "i8", description="Thick end position"),
        make_field_def("itemRgb", "itemRgb", description="Display"),
        make_field_def("blockCount", "blockCount", description="Number of blocks"),
        make_field_def("blockSizes", "blockSizes", description="Block sizes"),
        make_field_def(
            "blockStarts", "blockStarts", description="Block start positions"
        ),
    ]

    return fields[:num_fields]


def mkfields(bed_type):
    mandatory = mandatory_bed_field_definitions()
    optional = optional_bed_field_definitions(bed_type.value - BedType.BED3.value)
    return mandatory + optional


def mkschema(bed_path, out):
    bed_type = guess_bed_file_type(bed_path)
    data = pd.read_table(bed_path, header=None)
    spec = BedZarrSchema.generate(
        data.shape[0],
        bed_type,
    )
    out.write(spec.asjson())


def bed2zarr_init(
    bed_path,
    bed_type,
    zarr_path,
    *,
    schema_path=None,
    records_chunk_size=None,
):
    data = pd.read_table(bed_path, header=None)

    if schema_path is None:
        schema = BedZarrSchema.generate(
            data.shape[0],
            bed_type,
            records_chunk_size=records_chunk_size,
        )
    else:
        logger.info("Reading schema from %s", schema_path)
        if records_chunk_size is not None:
            raise ValueError(
                "Cannot specify schema along with chunk sizes"
            )  # NEEDS TEST
        with open(schema_path) as f:
            schema = BedZarrSchema.fromjson(f.read())

    # 2. init store
    bedzw = BedZarrWriter(zarr_path)
    bedzw.init(
        data,
        bed_path=bed_path,
        bed_type=bed_type,
        schema=schema,
    )
    return bedzw


def bed2zarr(
    bed_path,
    zarr_path,
    schema_path=None,
    records_chunk_size=None,
):
    bed_type = guess_bed_file_type(bed_path)
    bedzw = bed2zarr_init(
        bed_path,
        bed_type,
        zarr_path,
        schema_path=schema_path,
        records_chunk_size=records_chunk_size,
    )
    bedzw.write()
    bedzw.finalise()
