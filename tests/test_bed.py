import pytest
import zarr
from bio2zarr import bed2zarr


class TestBed:
    bed_path = "tests/data/bed/sample_mask.bed.gz"

    @pytest.fixture()
    def zarr(self, tmp_path):
        out = tmp_path / "sample_mask.zarr"
        return out

    def test_bed2zarr(self, zarr):
        bed2zarr.bed2zarr(bed_path=self.bed_path, zarr_path=zarr, show_progress=True)


class Test1kgBed:
    bed_path = "tests/data/bed/1kg_2020_chr20_annotations_mask.bed.gz"

    @pytest.fixture()
    def zarr(self, tmp_path):
        out = tmp_path / "1kg_2020_chr20_annotations_mask.zarr"
        return out

    def test_bed2zarr(self, zarr):
        bed2zarr.bed2zarr(bed_path=self.bed_path, zarr_path=zarr, show_progress=True)
