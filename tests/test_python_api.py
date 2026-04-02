import sys

import numpy as np
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


@pytest.mark.skipif(IS_WINDOWS, reason="VCF support requires cyvcf2")
class TestVcfConvertMode:
    vcf_path = "tests/data/vcf/sample.vcf.gz"

    def test_default_mode_is_read_only(self):
        root = vcf.convert([self.vcf_path])
        assert root.read_only

    def test_mode_r_is_read_only(self):
        root = vcf.convert([self.vcf_path], mode="r")
        assert root.read_only

    def test_mode_rplus_is_writable(self):
        root = vcf.convert([self.vcf_path], mode="r+")
        assert not root.read_only
        root.create_array("test", data=np.array([1, 2, 3]))
        assert "test" in root

    def test_mode_rplus_directory(self, tmp_path):
        zarr_path = tmp_path / "sample.vcz"
        root = vcf.convert([self.vcf_path], zarr_path, mode="r+")
        assert not root.read_only
        root.create_array("test", data=np.array([1, 2, 3]))
        assert "test" in root

    def test_encode_mode_rplus(self, tmp_path):
        icf_path = tmp_path / "icf"
        vcf.explode(icf_path, [self.vcf_path])
        root = vcf.encode(icf_path, mode="r+")
        assert not root.read_only
        root.create_array("test", data=np.array([1, 2, 3]))
        assert "test" in root


class TestPlinkConvertMode:
    plink_path = "tests/data/plink/example"

    def test_default_mode_is_read_only(self):
        root = plink.convert(self.plink_path)
        assert root.read_only

    def test_mode_rplus_is_writable(self):
        root = plink.convert(self.plink_path, mode="r+")
        assert not root.read_only
        root.create_array("test", data=np.array([1, 2, 3]))
        assert "test" in root

    def test_mode_rplus_directory(self, tmp_path):
        zarr_path = tmp_path / "sample.vcz"
        root = plink.convert(self.plink_path, zarr_path, mode="r+")
        assert not root.read_only
        root.create_array("test", data=np.array([1, 2, 3]))
        assert "test" in root


class TestTskitConvertMode:
    ts_path = "tests/data/tskit/example.trees"

    def test_default_mode_is_read_only(self):
        root = tskit_mod.convert(self.ts_path)
        assert root.read_only

    def test_mode_rplus_is_writable(self):
        root = tskit_mod.convert(self.ts_path, mode="r+")
        assert not root.read_only
        root.create_array("test", data=np.array([1, 2, 3]))
        assert "test" in root

    def test_mode_rplus_directory(self, tmp_path):
        zarr_path = tmp_path / "sample.vcz"
        root = tskit_mod.convert(self.ts_path, zarr_path, mode="r+")
        assert not root.read_only
        root.create_array("test", data=np.array([1, 2, 3]))
        assert "test" in root
