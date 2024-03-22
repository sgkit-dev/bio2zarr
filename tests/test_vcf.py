import json

import pytest


from bio2zarr import vcf


@pytest.fixture(scope="module")
def vcf_file():
    return "tests/data/vcf/sample.vcf.gz"


@pytest.fixture(scope="module")
def exploded_path(vcf_file, tmp_path_factory):
    out = tmp_path_factory.mktemp("data") / "example.exploded"
    vcf.explode([vcf_file], out)
    return out


@pytest.fixture(scope="module")
def schema_path(exploded_path, tmp_path_factory):
    out = tmp_path_factory.mktemp("data") / "example.schema.json"
    with open(out, "w") as f:
        vcf.mkschema(exploded_path, f)
    return out


@pytest.fixture(scope="module")
def schema(schema_path):
    with open(schema_path) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def zarr_path(exploded_path, tmp_path_factory):
    out = tmp_path_factory.mktemp("data") / "example.zarr"
    vcf.encode(exploded_path, out)
    return out


class TestJsonVersions:
    @pytest.mark.parametrize("version", ["0.1", "1.0", "xxxxx", 0.2])
    def test_zarr_schema_mismatch(self, schema, version):
        d = dict(schema)
        d["format_version"] = version
        with pytest.raises(ValueError, match="Zarr schema format version mismatch"):
            vcf.VcfZarrSchema.fromdict(d)

    @pytest.mark.parametrize("version", ["0.0", "1.0", "xxxxx", 0.1])
    def test_exploded_metadata_mismatch(self, tmpdir, exploded_path, version):
        with open(exploded_path / "metadata.json", "r") as f:
            d = json.load(f)

        d["format_version"] = version
        with pytest.raises(
            ValueError, match="Exploded metadata format version mismatch"
        ):
            vcf.VcfMetadata.fromdict(d)


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
            "filters": [{"id": "packbits"}],
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
            "filters": [{"id": "packbits"}],
        }
