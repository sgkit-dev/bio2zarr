import dataclasses
import numpy as np
import pathlib
import gzip
import zarr
from . import core


# see https://samtools.github.io/hts-specs/BEDv1.pdf
@dataclasses.dataclass
class Bed3:
    """BED3 genomic region with chromosome, start, and end. Intervals
    are 0-based, half-open."""

    chrom: str
    start: int
    end: int

    @property
    def width(self):
        """Width of the region."""
        return self.end - self.start

    def __len__(self):
        return self.width

    def mask(self, invert=False):
        """Create a mask for the region. The mask is an array of 1's
        (0's if inverted)."""
        func = np.zeros if invert else np.ones
        return func(self.width, dtype=np.uint8)


class BedReader:
    def __init__(self, bed_path):
        self.bed_path = pathlib.Path(bed_path)

    def __enter__(self):
        if self.bed_path.suffix == ".gz":
            self.fh = gzip.open(self.bed_path, "rt")
        else:
            self.fh = self.bed_path.open("rt")

        return self

    def __exit__(self, *args):
        self.fh.close()


# Here we are assuming that we write a mask. However, the BED file
# could represent other things, such as scores, and there could be up
# to 9 columns, in which case more fields (aka data arrays?) would be
# needed.
def bed2zarr(
    bed_path,
    zarr_path,
    bed_array="bed_mask",  # More generic name?
    show_progress=False,
):
    # 1. Make sure the bed file is gzipped and indexed
    bed_path = pathlib.Path(bed_path)

    if bed_path.suffix != ".gz":
        raise ValueError("BED file must be gzipped.")
    if (
        not bed_path.with_suffix(".gz.csi").exists()
        or not bed_path.with_suffix(".gz.tbi").exists()
    ):
        raise ValueError("BED file must be indexed.")

    # 2. Make sure there are contig lengths
    store = zarr.open(zarr_path)
    if "contig_length" not in store:
        raise ValueError(
            (
                "No contig lengths in Zarr store. Contig lengths must be"
                " present in the Zarr store before writing Bed entries."
            )
        )
    # 2b. Make chromosome to integer mapping
    chrom_d = {
        k: v for k, v in zip(store["contig_id"], np.arange(len(store["contig_id"])))
    }
    # 2c. Make cumulative index of contig lengths
    contig_indices = np.insert(np.cumsum(store["contig_length"])[:-1], 0, 0)

    # 3. Init the zarr group with the contig lengths
    # bed_array and bed_array_contig are of equal lengths = total genome
    if bed_array not in store:
        bed_array_contig = f"{bed_array}_contig"
        dtype = core.min_int_dtype(0, len(store["contig_id"]))
        n_bases = np.sum(store["contig_length"])

        store.create_dataset(bed_array, fill_value=0, dtype=dtype, shape=(n_bases,))
        store.create_dataset(
            bed_array_contig,
            data=np.repeat(
                np.arange(len(store["contig_id"])), store["contig_length"]
            ).astype(dtype),
            dtype=dtype,
            shape=(n_bases,),
        )

    # 4. Read the bed file and write the mask to the zarr dataset,
    # updating for each entry; many I/O operations; better read entire
    # file, store regions by chromosomes and generate index by
    # chromosome for all regions?
    with BedReader(bed_path) as br:
        for line in br.fh:
            chrom, start, end = line.strip().split("\t")
            i = chrom_d[chrom]
            start = int(start) + contig_indices[i]
            end = int(end) + contig_indices[i]
            bed = Bed3(chrom, start, end)
            mask = bed.mask()
            store[bed_array][start:end] = mask
