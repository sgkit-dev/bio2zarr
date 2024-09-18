import dataclasses
import logging
import pathlib

import numcodecs
import pandas as pd
import zarr

from . import core, provenance

logger = logging.getLogger(__name__)

DEFAULT_ZARR_COMPRESSOR = numcodecs.Blosc(cname="zstd", clevel=7)


@dataclasses.dataclass
class BedMetadata(core.JsonDataclass):
    fields: list


@dataclasses.dataclass
class BedField:
    category: str
    name: str


class BedZarrWriter:
    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.metadata = None
        self.data = None

    def init(
        self,
        bed_path,
        *,
        show_progress=False,
    ):
        self.read(bed_path)
        fields = mandatory_bed_field_definitions()
        # FIXME: add optional fields; depends on the number of columns
        # in the bed file. BED3 is the minimum.
        self.metadata = BedMetadata(fields)
        self.path.mkdir()
        store = zarr.DirectoryStore(self.path)
        root = zarr.group(store=store)
        root.attrs.update(
            {
                "bed_zarr_version": "0.1",
                "source": f"bio2zarr-{provenance.__version__}",
            }
        )
        self.encode_mandatory_fields(root)

    # FIXME: error checking, number of columns, etc.
    def read(self, bed_path):
        logger.info("Reading bed file %s", bed_path)
        first = pd.read_table(bed_path, header=None, nrows=1)
        num_cols = len(first.columns)
        if num_cols < 3:
            raise ValueError(f"Expected at least 3 columns in bed file, got {num_cols}")

        # FIXME: support chunked reading
        self.data = pd.read_table(bed_path, header=None).rename(
            columns={0: "chrom", 1: "chromStart", 2: "chromEnd"}
        )

    def encode_mandatory_fields(self, root):
        for field, dtype in zip(
            ["chrom", "chromStart", "chromEnd"], ["str", "int", "int"]
        ):
            # FIXME: Check schema for chunks
            array = root.array(
                field,
                self.data[field].values,
                dtype=dtype,
                compressor=DEFAULT_ZARR_COMPRESSOR,
                chunks=(1000,),
            )
            logger.debug("%s done", field)


# See https://samtools.github.io/hts-specs/BEDv1.pdf
def mandatory_bed_field_definitions():
    def make_field_def(name):
        return BedField(
            category="mandatory",
            name=name,
        )

    fields = [
        make_field_def("chrom"),
        make_field_def("chromStart"),
        make_field_def("chromEnd"),
    ]
    return fields


def bed2zarr(
    bed_path,
    zarr_path,
    show_progress=False,
):
    writer = BedZarrWriter(zarr_path)
    writer.init(bed_path, show_progress=show_progress)
