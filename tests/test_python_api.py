import sys

import pytest
import zarr

from bio2zarr import plink, vcf
from bio2zarr import tskit as tskit_mod

EXPECTED_VCF_ARRAYS = ["variant_position", "sample_id", "call_genotype"]
EXPECTED_PLINK_ARRAYS = ["variant_position", "sample_id", "call_genotype"]
EXPECTED_TSKIT_ARRAYS = ["variant_position", "sample_id", "call_genotype"]

IS_WINDOWS = sys.platform == "win32"


@pytest.mark.skipif(IS_WINDOWS, reason="VCF support requires cyvcf2")
class TestVcfConvert:
    vcf_path = "tests/data/vcf/sample.vcf.gz"

    def test_memory(self):
        root = vcf.convert([self.vcf_path])
        assert isinstance(root.store, zarr.storage.MemoryStore)
        for name in EXPECTED_VCF_ARRAYS:
            assert name in root

    def test_directory(self, tmp_path):
        zarr_path = tmp_path / "sample.vcz"
        root = vcf.convert([self.vcf_path], zarr_path)
        assert zarr_path.exists()
        for name in EXPECTED_VCF_ARRAYS:
            assert name in root

    def test_zip(self, tmp_path):
        zip_path = tmp_path / "sample.vcz.zip"
        root = vcf.convert([self.vcf_path], zip_path)
        assert zip_path.exists()
        assert isinstance(root.store, zarr.storage.ZipStore)
        for name in EXPECTED_VCF_ARRAYS:
            assert name in root

    def test_encode_memory(self, tmp_path):
        icf_path = tmp_path / "icf"
        vcf.explode(icf_path, [self.vcf_path])
        root = vcf.encode(icf_path)
        assert isinstance(root.store, zarr.storage.MemoryStore)
        for name in EXPECTED_VCF_ARRAYS:
            assert name in root


class TestPlinkConvert:
    plink_path = "tests/data/plink/example"

    def test_memory(self):
        root = plink.convert(self.plink_path)
        assert isinstance(root.store, zarr.storage.MemoryStore)
        for name in EXPECTED_PLINK_ARRAYS:
            assert name in root

    def test_directory(self, tmp_path):
        zarr_path = tmp_path / "sample.vcz"
        root = plink.convert(self.plink_path, zarr_path)
        assert zarr_path.exists()
        for name in EXPECTED_PLINK_ARRAYS:
            assert name in root

    def test_zip(self, tmp_path):
        zip_path = tmp_path / "sample.vcz.zip"
        root = plink.convert(self.plink_path, zip_path)
        assert zip_path.exists()
        assert isinstance(root.store, zarr.storage.ZipStore)
        for name in EXPECTED_PLINK_ARRAYS:
            assert name in root


class TestTskitConvert:
    ts_path = "tests/data/tskit/example.trees"

    def test_memory(self):
        root = tskit_mod.convert(self.ts_path)
        assert isinstance(root.store, zarr.storage.MemoryStore)
        for name in EXPECTED_TSKIT_ARRAYS:
            assert name in root

    def test_directory(self, tmp_path):
        zarr_path = tmp_path / "sample.vcz"
        root = tskit_mod.convert(self.ts_path, zarr_path)
        assert zarr_path.exists()
        for name in EXPECTED_TSKIT_ARRAYS:
            assert name in root

    def test_zip(self, tmp_path):
        zip_path = tmp_path / "sample.vcz.zip"
        root = tskit_mod.convert(self.ts_path, zip_path)
        assert zip_path.exists()
        assert isinstance(root.store, zarr.storage.ZipStore)
        for name in EXPECTED_TSKIT_ARRAYS:
            assert name in root
