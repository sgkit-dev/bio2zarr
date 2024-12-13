import numpy as np
import pandas as pd
import pytest
import zarr

from bio2zarr import bed2zarr

ALLOWED_BED_FORMATS = [3, 4, 5, 6, 7, 8, 9, 12]
SUPPORTED_BED_FORMATS = [3, 4, 5, 6, 7, 8]
DISALLOWED_BED_FORMATS = [2, 10, 11, 13]
ALL_BED_FORMATS = ALLOWED_BED_FORMATS + DISALLOWED_BED_FORMATS


@pytest.fixture()
def bed_data(request):
    data = [
        [
            "chr22",
            1000,
            5000,
            "cloneA",
            960,
            "+",
            1000,
            5000,
            0,
            2,
            "567,488",
            "0,3512",
            "foo",
        ],
        [
            "chr22",
            2000,
            6000,
            "cloneB",
            900,
            "-",
            2000,
            6000,
            0,
            2,
            "433,399",
            "0,3601",
            "bar",
        ],
    ]
    return [row[0 : request.param] for row in data]


@pytest.fixture()
def bed_df(bed_data):
    return pd.DataFrame(bed_data)


@pytest.fixture()
def bed_path(bed_data, tmp_path):
    out = tmp_path / "sample.bed"
    with open(out, "w") as fh:
        for row in bed_data:
            fh.write("\t".join(map(str, row)) + "\n")
    return out


class TestBed2Zarr:
    @pytest.mark.parametrize("bed_data", SUPPORTED_BED_FORMATS, indirect=True)
    def test_bed2zarr(self, bed_path, bed_df, tmp_path, request):
        bedspec = request.node.callspec.params["bed_data"]
        bed_type = bed2zarr.BedType(bedspec)
        zarr_path = tmp_path / "test.zarr"
        bed2zarr.bed2zarr(bed_path=bed_path, zarr_path=zarr_path)
        root = zarr.open(zarr_path)
        np.testing.assert_array_equal(root["contig"][:], [0, 0])
        np.testing.assert_array_equal(root["contig_id"][:], ["chr22"])
        np.testing.assert_array_equal(root["start"][:], bed_df[1].values)
        np.testing.assert_array_equal(root["end"][:], bed_df[2].values)
        if bed_type.value >= bed2zarr.BedType.BED4.value:
            np.testing.assert_array_equal(root["name"][:], [0, 1])
            np.testing.assert_array_equal(root["name_id"][:], bed_df[3].values)
        if bed_type.value >= bed2zarr.BedType.BED5.value:
            np.testing.assert_array_equal(root["score"][:], bed_df[4].values)
        if bed_type.value >= bed2zarr.BedType.BED6.value:
            np.testing.assert_array_equal(root["strand"][:], bed_df[5].values)
        if bed_type.value >= bed2zarr.BedType.BED7.value:
            np.testing.assert_array_equal(root["thickStart"][:], bed_df[6].values)
        if bed_type.value >= bed2zarr.BedType.BED8.value:
            np.testing.assert_array_equal(root["thickEnd"][:], bed_df[7].values)


class TestBedData:
    @pytest.mark.parametrize("bed_data", ALL_BED_FORMATS, indirect=True)
    def test_guess_bed_type_from_path(self, bed_path, request):
        bedspec = request.node.callspec.params["bed_data"]
        match_str = rf"(got {bedspec}|prohibited|unsupported)"
        if bedspec not in SUPPORTED_BED_FORMATS:
            with pytest.raises(ValueError, match=match_str):
                bed2zarr.guess_bed_file_type(bed_path)
        else:
            bed_type = bed2zarr.guess_bed_file_type(bed_path)
            assert bed_type.value == bedspec

    @pytest.mark.parametrize("bed_data", SUPPORTED_BED_FORMATS, indirect=True)
    def test_bed_fields(self, bed_path, request):
        bedspec = request.node.callspec.params["bed_data"]
        fields = bed2zarr.mkfields(bed2zarr.BedType(bedspec))
        assert len(fields) == bedspec


class TestSampleMaskBed:
    bed_path = "tests/data/bed/sample_mask.bed.gz"

    @pytest.fixture()
    def zarr_path(self, tmp_path):
        out = tmp_path / "sample_mask.zarr"
        return out

    def test_bed2zarr(self, zarr_path):
        bed2zarr.bed2zarr(bed_path=self.bed_path, zarr_path=zarr_path)


class Test1kgBed:
    bed_path = "tests/data/bed/1kg_2020_chr20_annotations_mask.bed.gz"

    @pytest.fixture()
    def zarr_path(self, tmp_path):
        out = tmp_path / "1kg_2020_chr20_annotations_mask.zarr"
        return out

    def test_bed2zarr(self, zarr_path):
        bed2zarr.bed2zarr(bed_path=self.bed_path, zarr_path=zarr_path)
