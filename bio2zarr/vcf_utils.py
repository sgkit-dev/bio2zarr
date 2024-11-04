import contextlib
import gzip
import logging
import os
import pathlib
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import IO, Any

import cyvcf2
import humanfriendly
import numpy as np

from bio2zarr.typing import PathType

logger = logging.getLogger(__name__)

CSI_EXTENSION = ".csi"
TABIX_EXTENSION = ".tbi"
TABIX_LINEAR_INDEX_INTERVAL_SIZE = 1 << 14  # 16kb interval size


def ceildiv(a: int, b: int) -> int:
    """Safe integer ceil function"""
    return -(-a // b)


def get_file_offset(vfp: int) -> int:
    """Convert a block compressed virtual file pointer to a file offset."""
    address_mask = 0xFFFFFFFFFFFF
    return vfp >> 16 & address_mask


def read_bytes_as_value(f: IO[Any], fmt: str, nodata: Any | None = None) -> Any:
    """Read bytes using a `struct` format string and return the unpacked data value.

    Parameters
    ----------
    f : IO[Any]
        The IO stream to read bytes from.
    fmt : str
        A Python `struct` format string.
    nodata : Optional[Any], optional
        The value to return in case there is no further data in the stream,
        by default None

    Returns
    -------
    Any
        The unpacked data value read from the stream.
    """
    data = f.read(struct.calcsize(fmt))
    if not data:
        return nodata
    values = struct.Struct(fmt).unpack(data)
    assert len(values) == 1
    return values[0]


def read_bytes_as_tuple(f: IO[Any], fmt: str) -> Sequence[Any]:
    """Read bytes using a `struct` format string and return the unpacked data values.

    Parameters
    ----------
    f : IO[Any]
        The IO stream to read bytes from.
    fmt : str
        A Python `struct` format string.

    Returns
    -------
    Sequence[Any]
        The unpacked data values read from the stream.
    """
    data = f.read(struct.calcsize(fmt))
    return struct.Struct(fmt).unpack(data)


@dataclass
class Region:
    """
    A htslib style region, where coordinates are 1-based and inclusive.
    """

    contig: str
    start: int | None = None
    end: int | None = None

    def __post_init__(self):
        if self.start is not None:
            self.start = int(self.start)
            assert self.start > 0
        if self.end is not None:
            self.end = int(self.end)
            assert self.end >= self.start

    def __str__(self):
        s = f"{self.contig}"
        if self.start is not None:
            s += f":{self.start}-"
        if self.end is not None:
            s += str(self.end)
        return s

    # TODO add "parse" class methoda for when we accept regions
    # as input


@dataclass
class Chunk:
    cnk_beg: int
    cnk_end: int


@dataclass
class CSIBin:
    bin: int
    loffset: int
    chunks: Sequence[Chunk]


RECORD_COUNT_UNKNOWN = np.inf


@dataclass
class CSIIndex:
    min_shift: int
    depth: int
    aux: str
    bins: Sequence[Sequence[CSIBin]]
    record_counts: Sequence[int]
    n_no_coor: int

    def parse_vcf_aux(self):
        assert len(self.aux) > 0
        # The first 7 values form the Tabix header or something, but I don't
        # know how to interpret what's in there. The n_ref value doesn't seem
        # to correspond to the number of contigs at all anyway, so just
        # ignoring for now.
        # values = struct.Struct("<7i").unpack(self.aux[:28])
        # tabix_header = Header(*values, 0)
        names = self.aux[28:]
        # Convert \0-terminated names to strings
        sequence_names = [str(name, "utf-8") for name in names.split(b"\x00")[:-1]]
        return sequence_names

    def offsets(self) -> Any:
        pseudo_bin = bin_limit(self.min_shift, self.depth) + 1

        file_offsets = []
        contig_indexes = []
        positions = []
        for contig_index, bins in enumerate(self.bins):
            # bins may be in any order within a contig, so sort by loffset
            for bin in sorted(bins, key=lambda b: b.loffset):
                if bin.bin == pseudo_bin:
                    continue  # skip pseudo bins
                file_offset = get_file_offset(bin.loffset)
                position = get_first_locus_in_bin(self, bin.bin)
                file_offsets.append(file_offset)
                contig_indexes.append(contig_index)
                positions.append(position)

        return np.array(file_offsets), np.array(contig_indexes), np.array(positions)


def bin_limit(min_shift: int, depth: int) -> int:
    """Defined in CSI spec"""
    return ((1 << (depth + 1) * 3) - 1) // 7


def get_first_bin_in_level(level: int) -> int:
    return ((1 << level * 3) - 1) // 7


def get_level_size(level: int) -> int:
    return 1 << level * 3


def get_level_for_bin(csi: CSIIndex, bin: int) -> int:
    for i in range(csi.depth, -1, -1):
        if bin >= get_first_bin_in_level(i):
            return i
    raise ValueError(f"Cannot find level for bin {bin}.")  # pragma: no cover


def get_first_locus_in_bin(csi: CSIIndex, bin: int) -> int:
    level = get_level_for_bin(csi, bin)
    first_bin_on_level = get_first_bin_in_level(level)
    level_size = get_level_size(level)
    max_span = 1 << (csi.min_shift + 3 * csi.depth)
    return (bin - first_bin_on_level) * (max_span // level_size) + 1


def read_csi(file: PathType, storage_options: dict[str, str] | None = None) -> CSIIndex:
    """Parse a CSI file into a `CSIIndex` object.

    Parameters
    ----------
    file : PathType
        The path to the CSI file.

    Returns
    -------
    CSIIndex
        An object representing a CSI index.

    Raises
    ------
    ValueError
        If the file is not a CSI file.
    """
    with gzip.open(file) as f:
        magic = read_bytes_as_value(f, "4s")
        if magic != b"CSI\x01":
            raise ValueError("File not in CSI format.")

        min_shift, depth, l_aux = read_bytes_as_tuple(f, "<3i")
        aux = read_bytes_as_value(f, f"{l_aux}s", "")
        n_ref = read_bytes_as_value(f, "<i")

        pseudo_bin = bin_limit(min_shift, depth) + 1

        bins = []
        record_counts = []

        if n_ref > 0:
            for _ in range(n_ref):
                n_bin = read_bytes_as_value(f, "<i")
                seq_bins = []
                # Distinguish between counts that are zero because the sequence
                # isn't there, vs counts that aren't in the index.
                record_count = 0 if n_bin == 0 else RECORD_COUNT_UNKNOWN
                for _ in range(n_bin):
                    bin, loffset, n_chunk = read_bytes_as_tuple(f, "<IQi")
                    chunks = []
                    for _ in range(n_chunk):
                        chunk = Chunk(*read_bytes_as_tuple(f, "<QQ"))
                        chunks.append(chunk)
                    seq_bins.append(CSIBin(bin, loffset, chunks))

                    if bin == pseudo_bin:
                        assert len(chunks) == 2
                        n_mapped, n_unmapped = chunks[1].cnk_beg, chunks[1].cnk_end
                        record_count = n_mapped + n_unmapped
                bins.append(seq_bins)
                record_counts.append(record_count)

        n_no_coor = read_bytes_as_value(f, "<Q", 0)

        assert len(f.read(1)) == 0

        return CSIIndex(min_shift, depth, aux, bins, record_counts, n_no_coor)


@dataclass
class Header:
    n_ref: int
    format: int
    col_seq: int
    col_beg: int
    col_end: int
    meta: int
    skip: int
    l_nm: int


@dataclass
class TabixBin:
    bin: int
    chunks: Sequence[Chunk]


@dataclass
class TabixIndex:
    header: Header
    sequence_names: Sequence[str]
    bins: Sequence[Sequence[TabixBin]]
    linear_indexes: Sequence[Sequence[int]]
    record_counts: Sequence[int]
    n_no_coor: int

    def offsets(self) -> Any:
        # Combine the linear indexes into one stacked array
        linear_indexes = self.linear_indexes
        linear_index = np.hstack([np.array(li) for li in linear_indexes])

        # Create file offsets for each element in the linear index
        file_offsets = np.array([get_file_offset(vfp) for vfp in linear_index])

        # Calculate corresponding contigs and positions or each element in
        # the linear index
        contig_indexes = np.hstack(
            [np.full(len(li), i) for (i, li) in enumerate(linear_indexes)]
        )
        # positions are 1-based and inclusive
        positions = np.hstack(
            [
                np.arange(len(li)) * TABIX_LINEAR_INDEX_INTERVAL_SIZE + 1
                for li in linear_indexes
            ]
        )
        assert len(file_offsets) == len(contig_indexes)
        assert len(file_offsets) == len(positions)

        return file_offsets, contig_indexes, positions


def read_tabix(
    file: PathType, storage_options: dict[str, str] | None = None
) -> TabixIndex:
    """Parse a tabix file into a `TabixIndex` object.

    Parameters
    ----------
    file : PathType
        The path to the tabix file.

    Returns
    -------
    TabixIndex
        An object representing a tabix index.

    Raises
    ------
    ValueError
        If the file is not a tabix file.
    """
    with gzip.open(file) as f:
        magic = read_bytes_as_value(f, "4s")
        if magic != b"TBI\x01":
            raise ValueError("File not in Tabix format.")

        header = Header(*read_bytes_as_tuple(f, "<8i"))

        sequence_names = []
        bins = []
        linear_indexes = []
        record_counts = []

        if header.l_nm > 0:
            names = read_bytes_as_value(f, f"<{header.l_nm}s")
            # Convert \0-terminated names to strings
            sequence_names = [str(name, "utf-8") for name in names.split(b"\x00")[:-1]]

            for _ in range(header.n_ref):
                n_bin = read_bytes_as_value(f, "<i")
                seq_bins = []
                # Distinguish between counts that are zero because the sequence
                # isn't there, vs counts that aren't in the index.
                record_count = 0 if n_bin == 0 else RECORD_COUNT_UNKNOWN
                for _ in range(n_bin):
                    bin, n_chunk = read_bytes_as_tuple(f, "<Ii")
                    chunks = []
                    for _ in range(n_chunk):
                        chunk = Chunk(*read_bytes_as_tuple(f, "<QQ"))
                        chunks.append(chunk)
                    seq_bins.append(TabixBin(bin, chunks))

                    if bin == 37450:  # pseudo-bin, see section 5.2 of BAM spec
                        assert len(chunks) == 2
                        n_mapped, n_unmapped = chunks[1].cnk_beg, chunks[1].cnk_end
                        record_count = n_mapped + n_unmapped
                n_intv = read_bytes_as_value(f, "<i")
                linear_index = []
                for _ in range(n_intv):
                    ioff = read_bytes_as_value(f, "<Q")
                    linear_index.append(ioff)
                bins.append(seq_bins)
                linear_indexes.append(linear_index)
                record_counts.append(record_count)

        n_no_coor = read_bytes_as_value(f, "<Q", 0)

        assert len(f.read(1)) == 0

        return TabixIndex(
            header, sequence_names, bins, linear_indexes, record_counts, n_no_coor
        )


class VcfFileType(Enum):
    VCF = ".vcf"
    BCF = ".bcf"


class VcfIndexType(Enum):
    CSI = ".csi"
    TABIX = ".tbi"


class IndexedVcf(contextlib.AbstractContextManager):
    def __init__(self, vcf_path, index_path=None):
        self.vcf = None
        vcf_path = pathlib.Path(vcf_path)
        if not vcf_path.exists():
            raise FileNotFoundError(vcf_path)
        if index_path is None:
            index_path = vcf_path.with_suffix(
                vcf_path.suffix + VcfIndexType.TABIX.value
            )
            if not index_path.exists():
                index_path = vcf_path.with_suffix(
                    vcf_path.suffix + VcfIndexType.CSI.value
                )
                if not index_path.exists():
                    raise FileNotFoundError(
                        f"Cannot find .tbi or .csi file for {vcf_path}"
                    )
        else:
            index_path = pathlib.Path(index_path)

        self.vcf_path = vcf_path
        self.index_path = index_path
        self.file_type = None
        self.index_type = None

        if index_path.suffix == VcfIndexType.CSI.value:
            self.index_type = VcfIndexType.CSI
        elif index_path.suffix == VcfIndexType.TABIX.value:
            self.index_type = VcfIndexType.TABIX
            self.file_type = VcfFileType.VCF
        else:
            raise ValueError("Only .tbi or .csi indexes are supported.")

        self.vcf = cyvcf2.VCF(vcf_path)
        self.vcf.set_index(str(self.index_path))
        logger.debug(f"Loaded {vcf_path} with index {self.index_path}")
        self.sequence_names = None

        if self.index_type == VcfIndexType.CSI:
            # Determine the file-type based on the "aux" field.
            self.index = read_csi(self.index_path)
            self.file_type = VcfFileType.BCF
            if len(self.index.aux) > 0:
                self.file_type = VcfFileType.VCF
                self.sequence_names = self.index.parse_vcf_aux()
            else:
                self.sequence_names = self.vcf.seqnames
        else:
            self.index = read_tabix(self.index_path)
            self.sequence_names = self.index.sequence_names

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.vcf is not None:
            self.vcf.close()
            self.vcf = None
        return False

    def contig_record_counts(self):
        d = dict(zip(self.sequence_names, self.index.record_counts, strict=True))
        if self.file_type == VcfFileType.BCF:
            d = {k: v for k, v in d.items() if v > 0}
        return d

    def count_variants(self, region):
        return sum(1 for _ in self.variants(region))

    def variants(self, region):
        start = 1 if region.start is None else region.start
        for var in self.vcf(str(region)):
            # Need to filter because of indels overlapping the region
            if var.POS >= start:
                yield var

    def _filter_empty_and_refine(self, regions):
        """
        Return all regions in the specified list that have one or more records,
        and refine the start coordinate of the region to be the actual first coord.

        Because this is a relatively expensive operation requiring seeking around
        the file, we return the results as an iterator.
        """
        for region in regions:
            var = next(self.variants(region), None)
            if var is not None:
                region.start = var.POS
                yield region

    def partition_into_regions(
        self,
        num_parts: int | None = None,
        target_part_size: None | int | str = None,
    ):
        if num_parts is None and target_part_size is None:
            raise ValueError("One of num_parts or target_part_size must be specified")

        if num_parts is not None and target_part_size is not None:
            raise ValueError(
                "Only one of num_parts or target_part_size may be specified"
            )

        if num_parts is not None and num_parts < 1:
            raise ValueError("num_parts must be positive")

        if target_part_size is not None:
            if isinstance(target_part_size, int):
                target_part_size_bytes = target_part_size
            else:
                target_part_size_bytes = humanfriendly.parse_size(target_part_size)
            if target_part_size_bytes < 1:
                raise ValueError("target_part_size must be positive")

        # Calculate the desired part file boundaries
        file_length = os.stat(self.vcf_path).st_size
        if num_parts is not None:
            target_part_size_bytes = file_length // num_parts
        elif target_part_size_bytes is not None:
            num_parts = ceildiv(file_length, target_part_size_bytes)
        part_lengths = target_part_size_bytes * np.arange(num_parts, dtype=int)
        file_offsets, region_contig_indexes, region_positions = self.index.offsets()

        # Search the file offsets to find which indexes the part lengths fall at
        ind = np.searchsorted(file_offsets, part_lengths)

        # Drop any parts that are greater than the file offsets
        # (these will be covered by a region with no end)
        ind = np.delete(ind, ind >= len(file_offsets))

        # Drop any duplicates
        ind = np.unique(ind)

        # Calculate region contig and start for each index
        region_contigs = region_contig_indexes[ind]
        region_starts = region_positions[ind]

        # Build region query strings
        regions = []
        for i in range(len(region_starts)):
            contig = self.sequence_names[region_contigs[i]]
            start = region_starts[i]

            if i == len(region_starts) - 1:  # final region
                regions.append(Region(contig, start))
            else:
                next_contig = self.sequence_names[region_contigs[i + 1]]
                next_start = region_starts[i + 1]
                end = next_start - 1  # subtract one since positions are inclusive
                # print("next_start", next_contig, next_start)
                if next_contig == contig:  # contig doesn't change
                    regions.append(Region(contig, start, end))
                else:
                    # contig changes, so need two regions (or possibly more if any
                    # sequences were skipped)
                    regions.append(Region(contig, start))
                    for ri in range(region_contigs[i] + 1, region_contigs[i + 1]):
                        regions.append(Region(self.sequence_names[ri]))
                    if end >= 1:
                        regions.append(Region(next_contig, 1, end))

        # Add any sequences at the end that were not skipped
        for ri in range(region_contigs[-1] + 1, len(self.sequence_names)):
            if self.index.record_counts[ri] > 0:
                regions.append(Region(self.sequence_names[ri]))

        return self._filter_empty_and_refine(regions)
