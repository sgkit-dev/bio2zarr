import pytest
import numpy as np
import numpy.testing as nt

from bio2zarr import vcf


class TestSmallExample:
    data_path = "tests/data/vcf/sample.vcf.gz"

    # fmt: off
    columns = [
        'ALT', 'CHROM', 'FILTERS', 'FORMAT/DP', 'FORMAT/GQ',
        'FORMAT/GT', 'FORMAT/HQ', 'ID', 'INFO/AA', 'INFO/AC',
        'INFO/AF', 'INFO/AN', 'INFO/DB', 'INFO/DP', 'INFO/H2',
        'INFO/NS', 'POS', 'QUAL', 'REF'
    ]
    # fmt: on

    @pytest.fixture(scope="class")
    def pcvcf(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("data") / "example.exploded"
        return vcf.explode([self.data_path], out)

    def test_mkschema(self, tmp_path, pcvcf):
        schema_file = tmp_path / "schema.json"
        with open(schema_file, "w") as f:
            vcf.mkschema(pcvcf.path, f)
        with open(schema_file, "r") as f:
            schema1 = vcf.ZarrConversionSpec.fromjson(f.read())
        schema2 = vcf.ZarrConversionSpec.generate(pcvcf)
        assert schema1 == schema2

    def test_summary_table(self, pcvcf):
        data = pcvcf.summary_table()
        cols = [d["name"] for d in data]
        assert sorted(cols) == self.columns

    def test_inspect(self, pcvcf):
        assert pcvcf.summary_table() == vcf.inspect(pcvcf.path)

    def test_mapping_methods(self, pcvcf):
        assert len(pcvcf) == len(self.columns)
        assert pcvcf["ALT"] is pcvcf.columns["ALT"]
        assert list(iter(pcvcf)) == list(iter(pcvcf))

    def test_num_partitions(self, pcvcf):
        assert pcvcf.num_partitions == 3

    def test_num_records(self, pcvcf):
        assert pcvcf.num_records == 9

    def test_POS(self, pcvcf):
        nt.assert_array_equal(
            [v[0] for v in pcvcf["POS"].values],
            [111, 112, 14370, 17330, 1110696, 1230237, 1234567, 1235237, 10],
        )

    def test_REF(self, pcvcf):
        ref = ["A", "A", "G", "T", "A", "T", "G", "T", "AC"]
        assert pcvcf["REF"].values == ref

    def test_ALT(self, pcvcf):
        alt = [
            ["C"],
            ["G"],
            ["A"],
            ["A"],
            ["G", "T"],
            [],
            ["GA", "GAC"],
            [],
            ["A", "ATG", "C"],
        ]
        assert [list(v) for v in pcvcf["ALT"].values] == alt

    def test_INFO_NS(self, pcvcf):
        assert pcvcf["INFO/NS"].values == [None, None, 3, 3, 2, 3, 3, None, None]


class TestGeneratedFieldsExample:
    data_path = "tests/data/vcf/field_type_combos.vcf.gz"

    @pytest.fixture(scope="class")
    def pcvcf(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("data") / "example.exploded"
        # import sgkit
        # from sgkit.io.vcf import vcf_to_zarr
        # vcf_to_zarr(self.data_path, "tmp/fields.vcf.sg", fields=
        #         ["INFO/IS1", "INFO/IC2", "INFO/IS2", "INFO/ISR", "FORMAT/FS2"])
        # df = sgkit.load_dataset("tmp/fields.vcf.sg")
        # print(df["variant_IC2"])
        # print(df["variant_IC2"].values)
        return vcf.explode([self.data_path], out)

    @pytest.fixture(scope="class")
    def schema(self, pcvcf):
        return vcf.ZarrConversionSpec.generate(pcvcf)

    @pytest.mark.parametrize(
        ("name", "dtype", "shape"),
        [
            ("variant_II1", "i1", (208,)),
            ("variant_II2", "i2", (208, 2)),
            ("variant_IIA", "i2", (208, 2)),
            ("variant_IIR", "i2", (208, 3)),
            ("variant_IID", "i2", (208, 7)),
            ("variant_IF1", "f4", (208,)),
            ("variant_IF2", "f4", (208, 2)),
            ("variant_IFA", "f4", (208, 2)),
            ("variant_IFR", "f4", (208, 3)),
            ("variant_IFD", "f4", (208, 9)),
            ("variant_IC1", "U1", (208,)),
            ("variant_IC2", "U1", (208, 2)),
            ("variant_IS1", "O", (208,)),
            ("variant_IS2", "O", (208, 2)),
            ("call_FS2", "O", (208, 2, 2)),
            ("call_FC2", "U1", (208, 2, 2)),
        ],
    )
    def test_info_schemas(self, schema, name, dtype, shape):
        v = schema.columns[name]
        assert v.dtype == dtype
        assert tuple(v.shape) == shape

    def test_info_string1(self, pcvcf):
        non_missing = [v for v in pcvcf["INFO/IS1"].values if v is not None]
        assert non_missing[0] == "bc"
        assert non_missing[1] == "."

    def test_info_char1(self, pcvcf):
        non_missing = [v for v in pcvcf["INFO/IC1"].values if v is not None]
        assert non_missing[0] == "f"
        assert non_missing[1] == "."

    def test_info_string2(self, pcvcf):
        non_missing = [v for v in pcvcf["INFO/IS2"].values if v is not None]
        nt.assert_array_equal(non_missing[0], ["hij", "d"])
        nt.assert_array_equal(non_missing[1], [".", "d"])
        nt.assert_array_equal(non_missing[2], ["hij", "."])
        nt.assert_array_equal(non_missing[3], [".", "."])

    def test_format_string1(self, pcvcf):
        non_missing = [v for v in pcvcf["FORMAT/FS1"].values if v is not None]
        nt.assert_array_equal(non_missing[0], [["bc"], ["."]])

    def test_format_string2(self, pcvcf):
        non_missing = [v for v in pcvcf["FORMAT/FS2"].values if v is not None]
        nt.assert_array_equal(non_missing[0], [["bc", "op"], [".", "op"]])
        nt.assert_array_equal(non_missing[1], [["bc", "."], [".", "."]])
