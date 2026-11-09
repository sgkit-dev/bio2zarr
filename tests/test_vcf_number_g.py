import sys

import numpy as np
import numpy.testing as nt
import pytest

from bio2zarr import vcf as vcf_mod
from tests.utils import load_dataset

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="VCF support requires cyvcf2"
)


def test_info_number_g_without_gt(tmp_path):
    path = tmp_path / "info-number-g.vcf"
    path.write_text(
        "##fileformat=VCFv4.3\n"
        "##contig=<ID=1>\n"
        '##INFO=<ID=IG,Number=G,Type=Float,Description="Genotype values">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "1\t1\t.\tA\tC\t.\tPASS\tIG=1.5,.,3.5,4.5\n"
    )

    ds = load_dataset(vcf_mod.convert([path], worker_processes=0))

    assert ds.variant_IG.dims == ("variants", "genotypes")
    assert ds.variant_IG.shape == (1, 4)
    assert "ploidy" not in ds.sizes
    nt.assert_array_equal(ds.variant_IG.values[0, [0, 2]], [1.5, 3.5])
    assert ds.variant_IG.values[0, 3] == 4.5
    assert np.isnan(ds.variant_IG.values[0, 1])


def test_info_number_g_without_observed_values(tmp_path):
    path = tmp_path / "missing-info-number-g.vcf"
    path.write_text(
        "##fileformat=VCFv4.3\n"
        "##contig=<ID=1>\n"
        '##INFO=<ID=IG,Number=G,Type=Float,Description="Genotype values">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "1\t1\t.\tA\tC\t.\tPASS\tIG=.\n"
    )

    ds = load_dataset(vcf_mod.convert([path], worker_processes=0))

    assert ds.variant_IG.dims == ("variants",)
    assert np.isnan(ds.variant_IG.values[0])


def test_format_number_g_without_gt_uses_observed_cardinality_with_explicit_ploidy(
    tmp_path,
):
    path = tmp_path / "format-number-g.vcf"
    path.write_text(
        "##fileformat=VCFv4.3\n"
        "##contig=<ID=1>\n"
        '##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Likelihood">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample\n"
        "1\t1\t.\tA\tC\t.\tPASS\t.\tPL\t1,2,3\n"
    )

    ds = load_dataset(vcf_mod.convert([path], ploidy=4, worker_processes=0))

    assert ds.call_PL.dims == ("variants", "samples", "genotypes")
    assert ds.call_PL.shape == (1, 1, 3)
    assert "ploidy" not in ds.sizes
