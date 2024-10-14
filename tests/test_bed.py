import pandas as pd
import pytest

from bio2zarr import bed2zarr

ALLOWED_BED_FORMATS = [3, 4, 5, 6, 7, 8, 9, 12]
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
    out = tmp_path / "sample_mask.bed"
    with open(out, "w") as fh:
        for row in bed_data:
            fh.write("\t".join(map(str, row)) + "\n")
    return out


@pytest.fixture()
def schema_path(bed_path, tmp_path_factory):
    out = tmp_path_factory.mktemp("data") / "example.schema.json"
    with open(out, "w") as f:
        bed2zarr.mkschema(bed_path, f)
    return out


@pytest.fixture()
def schema(schema_path):
    with open(schema_path) as f:
        return bed2zarr.BedZarrSchema.fromjson(f.read())


class TestDefaultSchema:
    @pytest.mark.parametrize("bed_data", ALLOWED_BED_FORMATS, indirect=True)
    def test_format_version(self, schema):
        assert schema.format_version == bed2zarr.ZARR_SCHEMA_FORMAT_VERSION


class TestSchema:
    @pytest.mark.parametrize("bed_data", ALLOWED_BED_FORMATS, indirect=True)
    def test_generate_schema(self, bed_df, request):
        bedspec = request.node.callspec.params["bed_data"]
        bed_type = bed2zarr.BedType(bedspec)
        fields = bed2zarr.mkfields(bed_type)
        bed_df.columns = [f.name for f in fields]
        bed_df, contig_id, feature_id = bed2zarr.encode_categoricals(bed_df, bed_type)
        fields = bed2zarr.update_field_bounds(bed_type, bed_df)
        schema = bed2zarr.BedZarrSchema.generate(bed_df.shape[0], bed_type, fields)
        assert schema.bed_type == bed_type.name
        assert schema.records_chunk_size == 1000


class TestBedData:
    @pytest.mark.parametrize("bed_data", ALL_BED_FORMATS, indirect=True)
    def test_guess_bed_type_from_path(self, bed_path, request):
        bedspec = request.node.callspec.params["bed_data"]
        if bedspec in DISALLOWED_BED_FORMATS:
            with pytest.raises(ValueError):
                bed2zarr.guess_bed_file_type(bed_path)
        else:
            bed_type = bed2zarr.guess_bed_file_type(bed_path)
            assert bed_type.value == bedspec

    @pytest.mark.parametrize("bed_data", ALLOWED_BED_FORMATS, indirect=True)
    def test_bed_fields(self, bed_path, request):
        bedspec = request.node.callspec.params["bed_data"]
        fields = bed2zarr.mkfields(bed2zarr.BedType(bedspec))
        assert len(fields) == bedspec


class TestSampleMaskBed:
    bed_path = "tests/data/bed/sample_mask.bed.gz"

    @pytest.fixture()
    def zarr(self, tmp_path):
        out = tmp_path / "sample_mask.zarr"
        return out

    def test_bed2zarr(self, zarr):
        bed2zarr.bed2zarr(bed_path=self.bed_path, zarr_path=zarr)


class Test1kgBed:
    bed_path = "tests/data/bed/1kg_2020_chr20_annotations_mask.bed.gz"

    @pytest.fixture()
    def zarr(self, tmp_path):
        out = tmp_path / "1kg_2020_chr20_annotations_mask.zarr"
        return out

    def test_bed2zarr(self, zarr):
        bed2zarr.bed2zarr(bed_path=self.bed_path, zarr_path=zarr)


class TestContigRename:
    bed_path = "tests/data/bed/1kg_2020_chr20_annotations_mask.bed.gz"
    bed_data = [
        "chr21\t50000\t60000\texon",
        "chr21\t60000\t90000\tfive_prime_utr",
        "chr20\t50000\t60000\texon",
        "chr20\t60000\t80000\tintron",
    ]

    @pytest.fixture()
    def bed(self, tmp_path):
        out = tmp_path / "test.bed"
        with open(out, "w") as fh:
            for row in self.bed_data:
                fh.write(row + "\n")
        return out
    
    def test_bed2zarr(self, bed, tmp_path):
        zarr = tmp_path / "test.zarr"

        data = pd.read_table(bed, header=None)
        print(data)
        print(data.describe(percentiles=[]))
        bedzw = bed2zarr.BedZarrWriter(zarr)
        print(bedzw)
        #print(bedzw.schema)
        print(bedzw.data)

