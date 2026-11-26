import functools
import sys

import numpy as np
import numpy.testing as nt
import pytest
import zarr

from bio2zarr import bgen, plink, vcf
from bio2zarr import tskit as tskit_mod
from tests.test_bgen import write_bgen

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


@pytest.fixture(
    params=[
        pytest.param(
            "vcf",
            marks=pytest.mark.skipif(IS_WINDOWS, reason="VCF support requires cyvcf2"),
        ),
        "plink",
        "tskit",
        "bgen",
    ]
)
def convert(request, tmp_path):
    """
    The convert function for each input format, with the input bound.
    """
    if request.param == "vcf":
        return functools.partial(vcf.convert, ["tests/data/vcf/sample.vcf.gz"])
    if request.param == "plink":
        return functools.partial(plink.convert, "tests/data/plink/example")
    if request.param == "tskit":
        return functools.partial(tskit_mod.convert, "tests/data/tskit/example.trees")
    bgen_path = tmp_path / "input.bgen"
    write_bgen(bgen_path)
    return functools.partial(bgen.convert, bgen_path)


class TestForce:
    # Overwriting existing output from the Python API (#415)

    def assert_same_as_fresh_conversion(self, root, convert):
        expected = convert()
        for name in ["sample_id", "variant_position"]:
            nt.assert_array_equal(root[name][:], expected[name][:])

    def test_existing_directory_without_force(self, convert, tmp_path):
        out = tmp_path / "out.vcz"
        out.mkdir()
        with pytest.raises(ValueError, match="already exists"):
            convert(out)

    def test_rerun_with_force(self, convert, tmp_path):
        out = tmp_path / "out.vcz"
        convert(out)
        root = convert(out, force=True)
        self.assert_same_as_fresh_conversion(root, convert)

    def test_force_replaces_directory(self, convert, tmp_path):
        out = tmp_path / "out.vcz"
        out.mkdir()
        (out / "old").write_text("old")
        root = convert(out, force=True)
        assert not (out / "old").exists()
        self.assert_same_as_fresh_conversion(root, convert)

    def test_force_replaces_zip(self, convert, tmp_path):
        out = tmp_path / "out.vcz.zip"
        out.write_text("not a zip file")
        root = convert(out, force=True)
        assert isinstance(root.store, zarr.storage.ZipStore)
        self.assert_same_as_fresh_conversion(root, convert)

    def test_force_without_existing_output(self, convert, tmp_path):
        out = tmp_path / "out.vcz"
        root = convert(out, force=True)
        assert out.exists()
        self.assert_same_as_fresh_conversion(root, convert)

    def test_force_in_memory(self, convert):
        root = convert(force=True)
        assert isinstance(root.store, zarr.storage.MemoryStore)


def test_bgen_force_keeps_existing_output_on_failure(tmp_path):
    # BGEN writes to a temporary location and only replaces the existing
    # output once the new store is complete.
    path = tmp_path / "input.bgen"
    write_bgen(path)
    sample_file = tmp_path / "malformed.sample"
    sample_file.write_text("ID_1\n0\nNA\n")
    out = tmp_path / "out.vcz"
    out.mkdir()
    (out / "keep").write_text("existing data")
    with pytest.raises(ValueError, match="missing identifier"):
        bgen.convert(path, out, sample_file=sample_file, force=True)
    assert (out / "keep").read_text() == "existing data"
