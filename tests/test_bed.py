import pytest
import sgkit as sg
import xarray.testing as xt
import zarr
from bio2zarr import bed2zarr, vcf2zarr


class Test1kgBed:
    data_path = "tests/data/vcf/1kg_2020_chr20_annotations.bcf"
    bed_path = "tests/data/bed/1kg_2020_chr20_annotations_mask.bed.gz"
    csi_path = "tests/data/bed/1kg_2020_chr20_annotations_mask.bed.gz.csi"

    @pytest.fixture(scope="module")
    def icf(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("data") / "1kg_2020.exploded"
        vcf2zarr.explode(out, [self.data_path])
        return out

    @pytest.fixture(scope="module")
    def zarr(self, icf, tmp_path_factory):
        out = tmp_path_factory.mktemp("data") / "1kg_2020.zarr"
        vcf2zarr.encode(icf, out)
        return out

    def test_add_mask_chr20(self, zarr):
        bed2zarr.bed2zarr(bed_path=self.bed_path, zarr_path=zarr, show_progress=True)


class TestSampleBed:
    data_path = "tests/data/vcf/sample.bcf"
    bed_path = "tests/data/bed/sample_mask.bed.gz"
    csi_path = "tests/data/bed/sample_mask.bed.gz.csi"

    @pytest.fixture(scope="module")
    def icf(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("data") / "sample.exploded"
        vcf2zarr.explode(out, [self.data_path])
        return out

    @pytest.fixture(scope="module")
    def zarr(self, icf, tmp_path_factory):
        out = tmp_path_factory.mktemp("data") / "sample.zarr"
        vcf2zarr.encode(icf, out)
        return out

    def test_add_mask_sample(self, zarr):
        with pytest.raises(ValueError):
            bed2zarr.bed2zarr(
                bed_path=self.bed_path, zarr_path=zarr, show_progress=True
            )
