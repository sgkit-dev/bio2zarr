from typing import Any, Dict, Optional, Sequence, Union
import contextlib
import struct
import re
import pathlib
import itertools
from dataclasses import dataclass

import fsspec
import numpy as np
from cyvcf2 import VCF
import cyvcf2
import humanfriendly

from bio2zarr.typing import PathType
from bio2zarr.utils import (
    get_file_offset,
    open_gzip,
    read_bytes_as_tuple,
    read_bytes_as_value,
    ceildiv,
    get_file_length,
)


# TODO create a Region dataclass that will sort correctly, and has
# a str method that does the correct thing


def region_filter(variants, region=None):
    """Filter out variants that don't start in the given region."""
    if region is None:
        return variants
    else:
        start = get_region_start(region)
        return itertools.filterfalse(lambda v: v.POS < start, variants)


def get_region_start(region: str) -> int:
    """Return the start position of the region string."""
    if re.search(r":\d+-\d*$", region):
        contig, start_end = region.rsplit(":", 1)
        start, end = start_end.split("-")
    else:
        return 1
    return int(start)


def region_string(contig: str, start: int, end: Optional[int] = None) -> str:
    if end is not None:
        return f"{contig}:{start}-{end}"
    else:
        return f"{contig}:{start}-"

@dataclass
class Region:
    contig: str
    start: Optional[int] = None
    end: Optional[int]=None

    def __str__(self):
        s = f"{self.contig}"
        if self.start is not None:
            s += f":{self.start}-"
        if self.end is not None:
            s += str(self.end)
        return s

    # TODO add "parse" class method

def get_tabix_path(
    vcf_path: PathType, storage_options: Optional[Dict[str, str]] = None
) -> Optional[str]:
    url = str(vcf_path)
    storage_options = storage_options or {}
    tbi_path = url + TABIX_EXTENSION
    with fsspec.open(url, **storage_options) as openfile:
        fs = openfile.fs
        if fs.exists(tbi_path):
            return tbi_path
        else:
            return None


def get_csi_path(
    vcf_path: PathType, storage_options: Optional[Dict[str, str]] = None
) -> Optional[str]:
    url = str(vcf_path)
    storage_options = storage_options or {}
    csi_path = url + CSI_EXTENSION
    with fsspec.open(url, **storage_options) as openfile:
        fs = openfile.fs
        if fs.exists(csi_path):
            return csi_path
        else:
            return None


def read_index(
    index_path: PathType, storage_options: Optional[Dict[str, str]] = None
) -> Any:
    url = str(index_path)
    if url.endswith(TABIX_EXTENSION):
        return read_tabix(url, storage_options=storage_options)
    elif url.endswith(CSI_EXTENSION):
        return read_csi(url, storage_options=storage_options)
    else:
        raise ValueError("Only .tbi or .csi indexes are supported.")


def get_sequence_names(vcf_path: PathType, index: Any) -> Any:
    try:
        # tbi stores sequence names
        return index.sequence_names
    except AttributeError:
        # ... but csi doesn't, so fall back to the VCF header
        return VCF(vcf_path).seqnames


def partition_into_regions(
    vcf_path: PathType,
    *,
    index_path: Optional[PathType] = None,
    num_parts: Optional[int] = None,
    target_part_size: Union[None, int, str] = None,
    storage_options: Optional[Dict[str, str]] = None,
) -> Optional[Sequence[str]]:
    """
    Calculate genomic region strings to partition a compressed VCF or BCF file into roughly equal parts.

    A ``.tbi`` or ``.csi`` file is used to find BGZF boundaries in the compressed VCF file, which are then
    used to divide the file into parts.

    The number of parts can specified directly by providing ``num_parts``, or by specifying the
    desired size (in bytes) of each (compressed) part by providing ``target_part_size``.
    Exactly one of ``num_parts`` or ``target_part_size`` must be provided.

    Both ``num_parts`` and ``target_part_size`` serve as hints: the number of parts and their sizes
    may be more or less than these parameters.

    Parameters
    ----------
    vcf_path
        The path to the VCF file.
    index_path
        The path to the VCF index (``.tbi`` or ``.csi``), by default None. If not specified, the
        index path is constructed by appending the index suffix (``.tbi`` or ``.csi``) to the VCF path.
    num_parts
        The desired number of parts to partition the VCF file into, by default None
    target_part_size
        The desired size, in bytes, of each (compressed) part of the partitioned VCF, by default None.
        If the value is a string, it may be specified using standard abbreviations, e.g. ``100MB`` is
        equivalent to ``100_000_000``.
    storage_options:
        Any additional parameters for the storage backend (see ``fsspec.open``).

    Returns
    -------
    The region strings that partition the VCF file, or None if the VCF file should not be partitioned
    (so there is only a single partition).

    Raises
    ------
    ValueError
        If neither of ``num_parts`` or ``target_part_size`` has been specified.
    ValueError
        If both of ``num_parts`` and ``target_part_size`` have been specified.
    ValueError
        If either of ``num_parts`` or ``target_part_size`` is not a positive integer.
    """
    if num_parts is None and target_part_size is None:
        raise ValueError("One of num_parts or target_part_size must be specified")

    if num_parts is not None and target_part_size is not None:
        raise ValueError("Only one of num_parts or target_part_size may be specified")

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
    file_length = get_file_length(vcf_path, storage_options=storage_options)
    if num_parts is not None:
        target_part_size_bytes = file_length // num_parts
    elif target_part_size_bytes is not None:
        num_parts = ceildiv(file_length, target_part_size_bytes)
    # FIXME - changing semantics from sgkit version here.
    # if num_parts == 1:
    #     return None
    part_lengths = np.array([i * target_part_size_bytes for i in range(num_parts)])

    if index_path is None:
        index_path = get_tabix_path(vcf_path, storage_options=storage_options)
        if index_path is None:
            index_path = get_csi_path(vcf_path, storage_options=storage_options)
            if index_path is None:
                raise ValueError("Cannot find .tbi or .csi file.")

    # Get the file offsets from .tbi/.csi
    index = read_index(index_path, storage_options=storage_options)
    sequence_names = get_sequence_names(vcf_path, index)
    file_offsets, region_contig_indexes, region_positions = index.offsets()

    # Search the file offsets to find which indexes the part lengths fall at
    ind = np.searchsorted(file_offsets, part_lengths)

    # Drop any parts that are greater than the file offsets
    # (these will be covered by a region with no end)
    ind = np.delete(ind, ind >= len(file_offsets))  # type: ignore[no-untyped-call]

    # Drop any duplicates
    ind = np.unique(ind)  # type: ignore[no-untyped-call]

    # Calculate region contig and start for each index
    region_contigs = region_contig_indexes[ind]
    region_starts = region_positions[ind]

    # Build region query strings
    regions = []
    for i in range(len(region_starts)):
        contig = sequence_names[region_contigs[i]]
        start = region_starts[i]

        if i == len(region_starts) - 1:  # final region
            regions.append(region_string(contig, start))
        else:
            next_contig = sequence_names[region_contigs[i + 1]]
            next_start = region_starts[i + 1]
            end = next_start - 1  # subtract one since positions are inclusive
            if next_contig == contig:  # contig doesn't change
                regions.append(region_string(contig, start, end))
            else:
                # contig changes, so need two regions (or possibly more if any
                # sequences were skipped)
                regions.append(region_string(contig, start))
                for ri in range(region_contigs[i] + 1, region_contigs[i + 1]):
                    regions.append(sequence_names[ri])
                regions.append(region_string(next_contig, 1, end))

    # https://github.com/pystatgen/sgkit/issues/1200
    # Turns out we need this for correctness. It's just that the
    # tests aren't particularly comprehensive. There must be some way we can
    # detect stuff that's in the index, and not in the header?

    # Add any sequences at the end that were not skipped
    for ri in range(region_contigs[-1] + 1, len(sequence_names)):
        regions.append(sequence_names[ri])

    return regions


CSI_EXTENSION = ".csi"


@dataclass
class Chunk:
    cnk_beg: int
    cnk_end: int


@dataclass
class CSIBin:
    bin: int
    loffset: int
    chunks: Sequence[Chunk]


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


def read_csi(
    file: PathType, storage_options: Optional[Dict[str, str]] = None
) -> CSIIndex:
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
    with open_gzip(file, storage_options=storage_options) as f:
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
                record_count = -1
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


TABIX_EXTENSION = ".tbi"
TABIX_LINEAR_INDEX_INTERVAL_SIZE = 1 << 14  # 16kb interval size


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

        # Calculate corresponding contigs and positions or each element in the linear index
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
    file: PathType, storage_options: Optional[Dict[str, str]] = None
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
    with open_gzip(file, storage_options=storage_options) as f:
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
                record_count = -1
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



class IndexedVcf:
    def __init__(self, path, index_path=None):
        # for h in vcf.header_iter():
        #     print(h)
        # if index_path is None:
        #     index_path = get_tabix_path(vcf_path, storage_options=storage_options)
        #     if index_path is None:
        #         index_path = get_csi_path(vcf_path, storage_options=storage_options)
        #         if index_path is None:
        #             raise ValueError("Cannot find .tbi or .csi file.")
        self.vcf_path = path
        self.index_path = index_path
        self.file_type = None
        self.index_type = None
        if index_path.suffix == ".csi":
            self.index_type = "csi"
        elif index_path.suffix == ".tbi":
            self.index_type = "tabix"
            self.file_type = "vcf"
        else:
            raise ValueError("TODO")
        self.index = read_index(self.index_path)
        self.sequence_names = None
        if self.index_type == "csi":
            # Determine the file-type based on the "aux" field.
            self.file_type = "bcf"
            if len(self.index.aux) > 0:
                self.file_type = "vcf"
                self.sequence_names = self.index.parse_vcf_aux()
            else:
                with contextlib.closing(cyvcf2.VCF(path)) as vcf:
                    self.sequence_names = vcf.seqnames
        else:
            self.sequence_names = self.index.sequence_names

    def contig_record_counts(self):
        d = dict(zip(self.sequence_names, self.index.record_counts))
        if self.file_type == "bcf":
            d = {k: v for k, v in d.items() if v > 0}
        return d

    def partition_into_regions(
        self,
        num_parts: Optional[int] = None,
        target_part_size: Union[None, int, str] = None,
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
        file_length = get_file_length(self.vcf_path)
        if num_parts is not None:
            target_part_size_bytes = file_length // num_parts
        elif target_part_size_bytes is not None:
            num_parts = ceildiv(file_length, target_part_size_bytes)
        part_lengths = np.array([i * target_part_size_bytes for i in range(num_parts)])

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
                if next_contig == contig:  # contig doesn't change
                    regions.append(Region(contig, start, end))
                else:
                    # contig changes, so need two regions (or possibly more if any
                    # sequences were skipped)
                    regions.append(Region(contig, start))
                    for ri in range(region_contigs[i] + 1, region_contigs[i + 1]):
                        regions.append(self.sequence_names[ri])
                    regions.append(Region(next_contig, 1, end))

        # Add any sequences at the end that were not skipped
        for ri in range(region_contigs[-1] + 1, len(self.sequence_names)):
            if self.index.record_counts[ri] > 0:
                regions.append(Region(self.sequence_names[ri]))

        return regions
