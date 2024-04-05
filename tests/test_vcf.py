import json
import pytest
import xarray.testing as xt
import sgkit as sg

from bio2zarr import vcf, vcf_utils


@pytest.fixture(scope="module")
def vcf_file():
    return "tests/data/vcf/sample.vcf.gz"


@pytest.fixture(scope="module")
def icf_path(vcf_file, tmp_path_factory):
    out = tmp_path_factory.mktemp("data") / "example.exploded"
    vcf.explode(out, [vcf_file])
    return out


@pytest.fixture(scope="module")
def schema_path(icf_path, tmp_path_factory):
    out = tmp_path_factory.mktemp("data") / "example.schema.json"
    with open(out, "w") as f:
        vcf.mkschema(icf_path, f)
    return out


@pytest.fixture(scope="module")
def schema(schema_path):
    with open(schema_path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def zarr_path(icf_path, tmp_path_factory):
    out = tmp_path_factory.mktemp("data") / "example.zarr"
    vcf.encode(icf_path, out)
    return out


class TestEncodeMaxMemory:
    @pytest.mark.parametrize(
        ["arg", "expected"],
        [
            (1, 1),
            (100.01, 100.01),
            ("1k", 1000),
            ("1KiB", 1024),
            ("1K", 1000),
            ("1MiB", 1024**2),
            ("1GiB", 1024**3),
            (None, 2**63),
        ],
    )
    def test_parser(self, arg, expected):
        assert vcf.parse_max_memory(arg) == expected

    @pytest.mark.parametrize("max_memory", [-1, 0, 1, "100 bytes"])
    def test_not_enough_memory(self, tmp_path, icf_path, max_memory):
        zarr_path = tmp_path / "zarr"
        with pytest.raises(ValueError, match="Insufficient memory"):
            vcf.encode(icf_path, zarr_path, max_memory=max_memory)

    @pytest.mark.parametrize("max_memory", [135, 269])
    def test_not_enough_memory_for_two(
        self, tmp_path, icf_path, zarr_path, caplog, max_memory
    ):
        other_zarr_path = tmp_path / "zarr"
        with caplog.at_level("DEBUG"):
            vcf.encode(
                icf_path, other_zarr_path, max_memory=max_memory, worker_processes=2
            )
        # This isn't a particularly strong test, but oh well.
        assert "Wait: mem_required" in caplog.text
        ds1 = sg.load_dataset(zarr_path)
        ds2 = sg.load_dataset(other_zarr_path)
        xt.assert_equal(ds1, ds2)


class TestJsonVersions:
    @pytest.mark.parametrize("version", ["0.1", "1.0", "xxxxx", 0.2])
    def test_zarr_schema_mismatch(self, schema, version):
        d = dict(schema)
        d["format_version"] = version
        with pytest.raises(ValueError, match="Zarr schema format version mismatch"):
            vcf.VcfZarrSchema.fromdict(d)

    @pytest.mark.parametrize("version", ["0.0", "1.0", "xxxxx", 0.1])
    def test_exploded_metadata_mismatch(self, tmpdir, icf_path, version):
        with open(icf_path / "metadata.json", "r") as f:
            d = json.load(f)

        d["format_version"] = version
        with pytest.raises(
            ValueError, match="Intermediate columnar metadata format version mismatch"
        ):
            vcf.IcfMetadata.fromdict(d)


class TestDefaultSchema:
    def test_format_version(self, schema):
        assert schema["format_version"] == vcf.ZARR_SCHEMA_FORMAT_VERSION

    def test_chunk_size(self, schema):
        assert schema["samples_chunk_size"] == 1000
        assert schema["variants_chunk_size"] == 10000

    def test_dimensions(self, schema):
        assert schema["dimensions"] == [
            "variants",
            "samples",
            "ploidy",
            "alleles",
            "filters",
        ]

    def test_sample_id(self, schema):
        assert schema["sample_id"] == ["NA00001", "NA00002", "NA00003"]

    def test_contig_id(self, schema):
        assert schema["contig_id"] == ["19", "20", "X"]

    def test_contig_length(self, schema):
        assert schema["contig_length"] is None

    def test_filter_id(self, schema):
        assert schema["filter_id"] == ["PASS", "s50", "q10"]

    def test_variant_contig(self, schema):
        assert schema["columns"]["variant_contig"] == {
            "name": "variant_contig",
            "dtype": "i2",
            "shape": [9],
            "chunks": [10000],
            "dimensions": ["variants"],
            "description": "",
            "vcf_field": None,
            "compressor": {
                "id": "blosc",
                "cname": "zstd",
                "clevel": 7,
                "shuffle": 0,
                "blocksize": 0,
            },
            "filters": [],
        }

    def test_call_genotype(self, schema):
        assert schema["columns"]["call_genotype"] == {
            "name": "call_genotype",
            "dtype": "i1",
            "shape": [9, 3, 2],
            "chunks": [10000, 1000],
            "dimensions": ["variants", "samples", "ploidy"],
            "description": "",
            "vcf_field": None,
            "compressor": {
                "id": "blosc",
                "cname": "zstd",
                "clevel": 7,
                "shuffle": 2,
                "blocksize": 0,
            },
            "filters": [],
        }

    def test_call_genotype_mask(self, schema):
        assert schema["columns"]["call_genotype_mask"] == {
            "name": "call_genotype_mask",
            "dtype": "bool",
            "shape": [9, 3, 2],
            "chunks": [10000, 1000],
            "dimensions": ["variants", "samples", "ploidy"],
            "description": "",
            "vcf_field": None,
            "compressor": {
                "id": "blosc",
                "cname": "zstd",
                "clevel": 7,
                "shuffle": 2,
                "blocksize": 0,
            },
            "filters": [],
        }

    def test_call_genotype_phased(self, schema):
        assert schema["columns"]["call_genotype_mask"] == {
            "name": "call_genotype_mask",
            "dtype": "bool",
            "shape": [9, 3, 2],
            "chunks": [10000, 1000],
            "dimensions": ["variants", "samples", "ploidy"],
            "description": "",
            "vcf_field": None,
            "compressor": {
                "id": "blosc",
                "cname": "zstd",
                "clevel": 7,
                "shuffle": 2,
                "blocksize": 0,
            },
            "filters": [],
        }


@pytest.mark.parametrize(
    "regions",
    [
        # Overlapping partitions
        [("1", 100, 200), ("1", 150, 250)],
        # Overlap by one position
        [("1", 100, 201), ("1", 200, 300)],
        # Contained overlap
        [("1", 100, 300), ("1", 150, 250)],
        # Exactly equal
        [("1", 100, 200), ("1", 100, 200)],
    ],
)
def test_check_overlap(regions):
    partitions = [
        vcf.VcfPartition("", region=vcf_utils.Region(contig, start, end))
        for contig, start, end in regions
    ]
    with pytest.raises(ValueError, match="Multiple VCFs have the region"):
        vcf.check_overlap(partitions)
