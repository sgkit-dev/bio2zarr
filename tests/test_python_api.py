import functools
import sys

import numpy as np
import pytest
import xarray.testing as xt
import zarr

from bio2zarr import bgen, plink, vcf
from bio2zarr import tskit as tskit_mod
from tests.test_bgen import write_bgen
from tests.utils import load_dataset

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


class TestForce:
    # Overwriting existing output from the Python API (#415)
    vcf_path = "tests/data/vcf/sample.vcf.gz"

    @pytest.fixture(
        params=[
            pytest.param(
                "vcf",
                marks=pytest.mark.skipif(
                    IS_WINDOWS, reason="VCF support requires cyvcf2"
                ),
            ),
            "plink",
            "tskit",
            "bgen",
        ]
    )
    def input_format(self, request):
        return request.param

    @pytest.fixture
    def convert(self, input_format, tmp_path):
        """
        The convert function for the input format, with a valid input bound.
        """
        if input_format == "vcf":
            return functools.partial(vcf.convert, [self.vcf_path])
        if input_format == "plink":
            return functools.partial(plink.convert, "tests/data/plink/example")
        if input_format == "tskit":
            return functools.partial(
                tskit_mod.convert, "tests/data/tskit/example.trees"
            )
        bgen_path = tmp_path / "input.bgen"
        write_bgen(bgen_path)
        return functools.partial(bgen.convert, bgen_path)

    @pytest.fixture
    def convert_missing_input(self, input_format, tmp_path):
        """
        The convert function for the input format, with an input path that
        does not exist.
        """
        missing = tmp_path / "missing"
        if input_format == "vcf":
            return functools.partial(vcf.convert, [missing.with_suffix(".vcf.gz")])
        if input_format == "plink":
            return functools.partial(plink.convert, missing)
        if input_format == "tskit":
            return functools.partial(tskit_mod.convert, missing.with_suffix(".trees"))
        return functools.partial(bgen.convert, missing.with_suffix(".bgen"))

    def assert_same_as_fresh_conversion(self, root, convert):
        xt.assert_equal(load_dataset(root), load_dataset(convert()))

    def test_existing_directory_without_force(self, convert, tmp_path):
        out = tmp_path / "out.vcz"
        out.mkdir()
        with pytest.raises(ValueError, match="already exists"):
            convert(out)

    def test_existing_zip_without_force(self, convert, tmp_path):
        out = tmp_path / "out.vcz.zip"
        out.write_text("existing data")
        with pytest.raises(ValueError, match="already exists"):
            convert(out)
        assert out.read_text() == "existing data"

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

    def test_missing_input_keeps_existing_output(self, convert_missing_input, tmp_path):
        out = tmp_path / "out.vcz"
        out.mkdir()
        (out / "keep").write_text("existing data")
        with pytest.raises((FileNotFoundError, RuntimeError), match="missing"):
            convert_missing_input(out, force=True)
        assert (out / "keep").read_text() == "existing data"

    def test_bgen_failed_conversion_keeps_existing_output(self, tmp_path):
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

    @pytest.mark.skipif(IS_WINDOWS, reason="VCF support requires cyvcf2")
    def test_vcf_force_replaces_icf(self, tmp_path):
        icf_path = tmp_path / "icf"
        out = tmp_path / "out.vcz"
        vcf.convert([self.vcf_path], out, icf_path=icf_path)
        with pytest.raises(ValueError, match="ICF path already exists"):
            vcf.convert([self.vcf_path], tmp_path / "other.vcz", icf_path=icf_path)
        root = vcf.convert([self.vcf_path], out, icf_path=icf_path, force=True)
        assert (icf_path / "metadata.json").exists()
        self.assert_same_as_fresh_conversion(
            root, functools.partial(vcf.convert, [self.vcf_path])
        )

    @pytest.mark.skipif(IS_WINDOWS, reason="VCF support requires cyvcf2")
    def test_vcf_encode_force(self, tmp_path):
        icf_path = tmp_path / "icf"
        vcf.explode(icf_path, [self.vcf_path])
        out = tmp_path / "out.vcz"
        vcf.encode(icf_path, out)
        with pytest.raises(ValueError, match="already exists"):
            vcf.encode(icf_path, out)
        root = vcf.encode(icf_path, out, force=True)
        xt.assert_equal(load_dataset(root), load_dataset(vcf.encode(icf_path)))
