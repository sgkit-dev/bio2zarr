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
def zarr_path(exploded_path, tmp_path_factory):
    out = tmp_path_factory.mktemp("data") / "example.zarr"
    vcf.encode(exploded_path, out)
    return out


class TestJsonVersions:
    @pytest.mark.parametrize("version", ["0.1", "1.0", "xxxxx", 0.2])
    def test_zarr_schema_mismatch(self, schema_path, version):
        with open(schema_path) as f:
            d = json.load(f)

        d["format_version"] = version
        with pytest.raises(ValueError, match="Zarr schema format version mismatch"):
            vcf.ZarrConversionSpec.fromdict(d)

    @pytest.mark.parametrize("version", ["0.0", "1.0", "xxxxx", 0.1])
    def test_exploded_metadata_mismatch(self, tmpdir, exploded_path, version):
        with open(exploded_path / "metadata.json", "r") as f:
            d = json.load(f)

        d["format_version"] = version
        with pytest.raises(
            ValueError, match="Exploded metadata format version mismatch"
        ):
            vcf.VcfMetadata.fromdict(d)
