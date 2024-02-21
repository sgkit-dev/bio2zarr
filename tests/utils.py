from contextlib import contextmanager
import itertools
from pathlib import Path
import re
from typing import Iterator, Optional

from cyvcf2 import VCF, Variant

from sgkit.typing import PathType


def path_for_test(shared_datadir: Path, file: str, is_path: bool = True) -> PathType:
    """Return a test data path whose type is determined by `is_path`.

    If `is_path` is True, return a `Path`, otherwise return a `str`.
    """
    path: PathType = shared_datadir / file
    return path if is_path else str(path)


@contextmanager
def open_vcf(path: PathType) -> Iterator[VCF]:
    """A context manager for opening a VCF file."""
    vcf = VCF(path)
    try:
        yield vcf
    finally:
        vcf.close()


def region_filter(
    variants: Iterator[Variant], region: Optional[str] = None
) -> Iterator[Variant]:
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


def count_variants(path: PathType, region: Optional[str] = None) -> int:
    """Count the number of variants in a VCF file."""
    with open_vcf(path) as vcf:
        if region is not None:
            vcf = vcf(region)
        return sum(1 for _ in region_filter(vcf, region))