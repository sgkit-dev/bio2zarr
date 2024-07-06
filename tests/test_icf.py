import pickle
import shutil

import numcodecs
import numpy as np
import numpy.testing as nt
import pytest

from bio2zarr import provenance, vcf2zarr, vcf_utils
from bio2zarr.vcf2zarr import icf as icf_mod


class TestSmallExample:
    data_path = "tests/data/vcf/sample.vcf.gz"

    # fmt: off
    fields = (
        'ALT', 'CHROM', 'FILTERS', 'FORMAT/DP', 'FORMAT/GQ',
        'FORMAT/GT', 'FORMAT/HQ', 'ID', 'INFO/AA', 'INFO/AC',
        'INFO/AF', 'INFO/AN', 'INFO/DB', 'INFO/DP', 'INFO/H2',
        'INFO/NS', 'POS', 'QUAL', 'REF'
    )
    # fmt: on

    @pytest.fixture(scope="class")
    def icf(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("data") / "example.exploded"
        return vcf2zarr.explode(out, [self.data_path], local_alleles=False)

    def test_format_version(self, icf):
        assert icf.metadata.format_version == icf_mod.ICF_METADATA_FORMAT_VERSION

    def test_provenance(self, icf):
        assert icf.metadata.provenance == {
            "source": f"bio2zarr-{provenance.__version__}"
        }

    def test_mkschema(self, tmp_path, icf):
        schema_file = tmp_path / "schema.json"
        with open(schema_file, "w") as f:
            vcf2zarr.mkschema(icf.path, f)
        with open(schema_file) as f:
            schema1 = vcf2zarr.VcfZarrSchema.fromjson(f.read())
        schema2 = vcf2zarr.VcfZarrSchema.generate(icf)
        assert schema1 == schema2

    def test_summary_table(self, icf):
        data = icf.summary_table()
        fields = [d["name"] for d in data]
        assert tuple(sorted(fields)) == self.fields

    def test_inspect(self, icf):
        assert icf.summary_table() == vcf2zarr.inspect(icf.path)

    def test_mapping_methods(self, icf):
        assert len(icf) == len(self.fields)
        assert icf["ALT"] is icf.fields["ALT"]
        assert list(iter(icf)) == list(iter(icf))

    def test_num_partitions(self, icf):
        assert icf.num_partitions == 3

    def test_num_records(self, icf):
        assert icf.num_records == 9

    def test_POS(self, icf):
        nt.assert_array_equal(
            [v[0] for v in icf["POS"].values],
            [111, 112, 14370, 17330, 1110696, 1230237, 1234567, 1235237, 10],
        )

    def test_REF(self, icf):
        ref = ["A", "A", "G", "T", "A", "T", "G", "T", "AC"]
        assert icf["REF"].values == ref

    def test_ALT(self, icf):
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
        assert [list(v) for v in icf["ALT"].values] == alt

    def test_INFO_NS(self, icf):
        assert icf["INFO/NS"].values == [None, None, 3, 3, 2, 3, 3, None, None]


class TestLocalAllelesExample:
    data_path = "tests/data/vcf/local_alleles.vcf.gz"

    fields = (
        "ALT",
        "CHROM",
        "FILTERS",
        "FORMAT/AD",
        "FORMAT/DP",
        "FORMAT/GQ",
        "FORMAT/GT",
        "FORMAT/LAA",
        "FORMAT/PL",
        "ID",
        "INFO/AA",
        "INFO/AC",
        "INFO/AF",
        "INFO/AN",
        "INFO/DB",
        "INFO/DP",
        "INFO/H2",
        "INFO/NS",
        "POS",
        "QUAL",
        "REF",
    )

    @pytest.fixture(scope="class")
    def icf(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("data") / "example.exploded"
        return vcf2zarr.explode(out, [self.data_path])

    def test_summary_table(self, icf):
        data = icf.summary_table()
        fields = [d["name"] for d in data]
        assert tuple(sorted(fields)) == self.fields


class TestIcfWriterExample:
    data_path = "tests/data/vcf/sample.vcf.gz"

    # fmt: off
    fields = (
        'ALT', 'CHROM', 'FILTERS', 'FORMAT/DP', 'FORMAT/GQ',
        'FORMAT/GT', 'FORMAT/HQ', 'ID', 'INFO/AA', 'INFO/AC',
        'INFO/AF', 'INFO/AN', 'INFO/DB', 'INFO/DP', 'INFO/H2',
        'INFO/NS', 'POS', 'QUAL', 'REF'
    )
    # fmt: on

    def test_init_paths(self, tmp_path):
        icf_path = tmp_path / "x.icf"
        assert not icf_path.exists()
        summary = vcf2zarr.explode_init(icf_path, [self.data_path])
        assert summary.num_partitions == 3
        assert icf_path.exists()
        wip_path = icf_path / "wip"
        assert wip_path.exists()
        for field_name in self.fields:
            field_path = icf_path / field_name
            assert field_path.exists()
            assert field_path.is_dir()

    def test_finalise_paths(self, tmp_path):
        icf_path = tmp_path / "x.icf"
        wip_path = icf_path / "wip"
        summary = vcf2zarr.explode_init(icf_path, [self.data_path])
        assert icf_path.exists()
        for j in range(summary.num_partitions):
            vcf2zarr.explode_partition(icf_path, j)
        assert wip_path.exists()
        vcf2zarr.explode_finalise(icf_path)
        assert icf_path.exists()
        assert not wip_path.exists()

    def test_finalise_no_partitions_fails(self, tmp_path):
        icf_path = tmp_path / "x.icf"
        vcf2zarr.explode_init(icf_path, [self.data_path])
        with pytest.raises(FileNotFoundError, match="3 partitions: \\[0, 1, 2\\]"):
            vcf2zarr.explode_finalise(icf_path)

    @pytest.mark.parametrize("partition", [0, 1, 2])
    def test_finalise_missing_partition_fails(self, tmp_path, partition):
        icf_path = tmp_path / "x.icf"
        vcf2zarr.explode_init(icf_path, [self.data_path])
        for j in range(3):
            if j != partition:
                vcf2zarr.explode_partition(icf_path, j)
        with pytest.raises(FileNotFoundError, match=f"1 partitions: \\[{partition}\\]"):
            vcf2zarr.explode_finalise(icf_path)

    @pytest.mark.parametrize("partition", [0, 1, 2])
    def test_explode_partition(self, tmp_path, partition):
        icf_path = tmp_path / "x.icf"
        vcf2zarr.explode_init(icf_path, [self.data_path])
        summary_file = icf_path / "wip" / f"p{partition}.json"
        assert not summary_file.exists()
        vcf2zarr.explode_partition(icf_path, partition)
        assert summary_file.exists()

    def test_double_explode_partition(self, tmp_path):
        partition = 1
        icf_path = tmp_path / "x.icf"
        vcf2zarr.explode_init(icf_path, [self.data_path])
        summary_file = icf_path / "wip" / f"p{partition}.json"
        assert not summary_file.exists()
        vcf2zarr.explode_partition(icf_path, partition)
        with open(summary_file) as f:
            s1 = f.read()
        vcf2zarr.explode_partition(icf_path, partition)
        with open(summary_file) as f:
            s2 = f.read()
        assert s1 == s2

    @pytest.mark.parametrize("partition", [-1, 3, 100])
    def test_explode_partition_out_of_range(self, tmp_path, partition):
        icf_path = tmp_path / "x.icf"
        vcf2zarr.explode_init(icf_path, [self.data_path])
        with pytest.raises(ValueError, match="Partition index not in the valid range"):
            vcf2zarr.explode_partition(icf_path, partition)

    def test_explode_same_file_twice(self, tmp_path):
        icf_path = tmp_path / "x.icf"
        with pytest.raises(ValueError, match="Duplicate path provided"):
            vcf2zarr.explode(icf_path, [self.data_path, self.data_path])

    def test_explode_same_data_twice(self, tmp_path):
        icf_path = tmp_path / "x.icf"
        with pytest.raises(ValueError, match="Overlapping VCF regions"):
            vcf2zarr.explode(icf_path, [self.data_path, "tests/data/vcf/sample.bcf"])


class TestGeneratedFieldsExample:
    data_path = "tests/data/vcf/field_type_combos.vcf.gz"

    @pytest.fixture(scope="class")
    def icf(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("data") / "example.exploded"
        # import sgkit
        # from sgkit.io.vcf import vcf_to_zarr
        # vcf_to_zarr(self.data_path, "tmp/fields.vcf.sg", fields=
        #         ["INFO/IS1", "INFO/IC2", "INFO/IS2", "INFO/ISR", "FORMAT/FS2"])
        # df = sgkit.load_dataset("tmp/fields.vcf.sg")
        # print(df["variant_IC2"])
        # print(df["variant_IC2"].values)
        return vcf2zarr.explode(out, [self.data_path])

    @pytest.fixture(scope="class")
    def schema(self, icf):
        return vcf2zarr.VcfZarrSchema.generate(icf)

    @pytest.mark.parametrize(
        ("name", "dtype", "shape", "dimensions"),
        [
            ("variant_II1", "i1", (208,), ("variants",)),
            ("variant_II2", "i2", (208, 2), ("variants", "INFO_II2_dim")),
            ("variant_IIA", "i2", (208, 2), ("variants", "alt_alleles")),
            ("variant_IIR", "i2", (208, 3), ("variants", "alleles")),
            ("variant_IID", "i2", (208, 7), ("variants", "INFO_IID_dim")),
            ("variant_IF1", "f4", (208,), ("variants",)),
            ("variant_IF2", "f4", (208, 2), ("variants", "INFO_IF2_dim")),
            ("variant_IFA", "f4", (208, 2), ("variants", "alt_alleles")),
            ("variant_IFR", "f4", (208, 3), ("variants", "alleles")),
            ("variant_IFD", "f4", (208, 9), ("variants", "INFO_IFD_dim")),
            ("variant_IC1", "U1", (208,), ("variants",)),
            ("variant_IC2", "U1", (208, 2), ("variants", "INFO_IC2_dim")),
            ("variant_IS1", "O", (208,), ("variants",)),
            ("variant_IS2", "O", (208, 2), ("variants", "INFO_IS2_dim")),
            ("call_FS2", "O", (208, 2, 2), ("variants", "samples", "FORMAT_FS2_dim")),
            ("call_FC2", "U1", (208, 2, 2), ("variants", "samples", "FORMAT_FC2_dim")),
            ("call_FIG", "i2", (208, 2, 6), ("variants", "samples", "genotypes")),
            ("call_FIA", "i2", (208, 2, 2), ("variants", "samples", "alt_alleles")),
            ("call_FIR", "i2", (208, 2, 3), ("variants", "samples", "alleles")),
        ],
    )
    def test_info_schemas(self, schema, name, dtype, shape, dimensions):
        v = schema.field_map()[name]
        assert v.dtype == dtype
        assert tuple(v.shape) == shape
        assert v.dimensions == dimensions

    def test_info_string1(self, icf):
        non_missing = [v for v in icf["INFO/IS1"].values if v is not None]
        assert non_missing[0] == "bc"
        assert non_missing[1] == "."

    def test_info_char1(self, icf):
        non_missing = [v for v in icf["INFO/IC1"].values if v is not None]
        assert non_missing[0] == "f"
        assert non_missing[1] == "."

    def test_info_string2(self, icf):
        non_missing = [v for v in icf["INFO/IS2"].values if v is not None]
        nt.assert_array_equal(non_missing[0], ["hij", "d"])
        nt.assert_array_equal(non_missing[1], [".", "d"])
        nt.assert_array_equal(non_missing[2], ["hij", "."])
        nt.assert_array_equal(non_missing[3], [".", "."])

    def test_format_string1(self, icf):
        non_missing = [v for v in icf["FORMAT/FS1"].values if v is not None]
        nt.assert_array_equal(non_missing[0], [["bc"], ["."]])

    def test_format_string2(self, icf):
        non_missing = [v for v in icf["FORMAT/FS2"].values if v is not None]
        nt.assert_array_equal(non_missing[0], [["bc", "op"], [".", "op"]])
        nt.assert_array_equal(non_missing[1], [["bc", "."], [".", "."]])


class TestInitProperties:
    data_path = "tests/data/vcf/sample.vcf.gz"

    def run_explode(self, tmp_path, **kwargs):
        icf_path = tmp_path / "icf"
        vcf2zarr.explode(icf_path, [self.data_path], **kwargs)
        return vcf2zarr.IntermediateColumnarFormat(icf_path)

    def run_dexplode(self, tmp_path, **kwargs):
        icf_path = tmp_path / "icf"
        summary = vcf2zarr.explode_init(icf_path, [self.data_path], **kwargs)
        for j in range(summary.num_partitions):
            vcf2zarr.explode_partition(icf_path, j)
        vcf2zarr.explode_finalise(icf_path)
        return vcf2zarr.IntermediateColumnarFormat(icf_path)

    @pytest.mark.parametrize(
        "compressor",
        [
            numcodecs.Blosc(),
            numcodecs.Blosc(cname="lz4", clevel=1, shuffle=numcodecs.Blosc.SHUFFLE),
            numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.NOSHUFFLE),
        ],
    )
    def test_compressor_explode(self, tmp_path, compressor):
        icf = self.run_explode(tmp_path, compressor=compressor)
        assert icf.metadata.compressor == compressor.get_config()

    def test_default_compressor_explode(self, tmp_path):
        icf = self.run_explode(tmp_path)
        assert icf.metadata.compressor == icf_mod.ICF_DEFAULT_COMPRESSOR.get_config()
        assert icf.metadata.compressor["cname"] == "zstd"

    def test_default_compressor_dexplode(self, tmp_path):
        icf = self.run_dexplode(tmp_path)
        assert icf.metadata.compressor == icf_mod.ICF_DEFAULT_COMPRESSOR.get_config()
        assert icf.metadata.compressor["cname"] == "zstd"

    @pytest.mark.parametrize(
        "compressor",
        [
            numcodecs.Blosc(),
            numcodecs.Blosc(cname="lz4", clevel=1, shuffle=numcodecs.Blosc.SHUFFLE),
            numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.NOSHUFFLE),
        ],
    )
    def test_compressor_init(self, tmp_path, compressor):
        icf = self.run_dexplode(tmp_path, compressor=compressor)
        assert icf.metadata.compressor == compressor.get_config()

    @pytest.mark.parametrize("column_chunk_size", [1, 10, 0.125])
    def test_column_chunk_size_explode(self, tmp_path, column_chunk_size):
        icf = self.run_explode(tmp_path, column_chunk_size=column_chunk_size)
        assert icf.metadata.column_chunk_size == column_chunk_size

    @pytest.mark.parametrize("column_chunk_size", [1, 10, 0.125])
    def test_column_chunk_size_dexplode(self, tmp_path, column_chunk_size):
        icf = self.run_dexplode(tmp_path, column_chunk_size=column_chunk_size)
        assert icf.metadata.column_chunk_size == column_chunk_size


class TestCorruptionDetection:
    data_path = "tests/data/vcf/sample.vcf.gz"

    def test_missing_field(self, tmp_path):
        icf_path = tmp_path / "icf"
        vcf2zarr.explode(icf_path, [self.data_path])
        shutil.rmtree(icf_path / "POS")
        icf = vcf2zarr.IntermediateColumnarFormat(icf_path)
        with pytest.raises(FileNotFoundError):
            icf["POS"].values  # noqa B018

    def test_missing_chunk_index(self, tmp_path):
        icf_path = tmp_path / "icf"
        vcf2zarr.explode(icf_path, [self.data_path])
        chunk_index_path = icf_path / "POS" / "p0" / "chunk_index"
        assert chunk_index_path.exists()
        chunk_index_path.unlink()
        icf = vcf2zarr.IntermediateColumnarFormat(icf_path)
        with pytest.raises(FileNotFoundError):
            icf["POS"].values  # noqa B018

    def test_missing_chunk_file(self, tmp_path):
        icf_path = tmp_path / "icf"
        vcf2zarr.explode(icf_path, [self.data_path])
        chunk_file = icf_path / "POS" / "p0" / "2"
        assert chunk_file.exists()
        chunk_file.unlink()
        icf = vcf2zarr.IntermediateColumnarFormat(icf_path)
        with pytest.raises(FileNotFoundError):
            icf["POS"].values  # noqa B018

    def test_empty_chunk_file(self, tmp_path):
        icf_path = tmp_path / "icf"
        vcf2zarr.explode(icf_path, [self.data_path])
        chunk_file = icf_path / "POS" / "p0" / "2"
        assert chunk_file.exists()
        with open(chunk_file, "w") as _:
            pass
        icf = vcf2zarr.IntermediateColumnarFormat(icf_path)
        with pytest.raises(RuntimeError, match="blosc"):
            icf["POS"].values  # noqa B018

    # Chunk file is 195 long for numpy v1, and 193 long for v2
    @pytest.mark.parametrize("length", [10, 100, 190, 192])
    def test_truncated_chunk_file(self, tmp_path, length):
        icf_path = tmp_path / "icf"
        vcf2zarr.explode(icf_path, [self.data_path])
        chunk_file = icf_path / "POS" / "p0" / "2"
        with open(chunk_file, "rb") as f:
            buff = f.read(length)
        assert len(buff) == length
        with open(chunk_file, "wb") as f:
            f.write(buff)
        icf = vcf2zarr.IntermediateColumnarFormat(icf_path)
        # Either Blosc or pickling errors happen here
        with pytest.raises((RuntimeError, pickle.UnpicklingError)):
            icf["POS"].values  # noqa B018

    def test_chunk_incorrect_length(self, tmp_path):
        icf_path = tmp_path / "icf"
        vcf2zarr.explode(icf_path, [self.data_path])
        chunk_file = icf_path / "POS" / "p0" / "2"
        compressor = numcodecs.Blosc(cname="zstd")
        with open(chunk_file, "rb") as f:
            pkl = compressor.decode(f.read())
        x = pickle.loads(pkl)
        assert len(x) == 2
        # Write back a chunk with the incorrect number of records
        pkl = pickle.dumps(x[0])
        with open(chunk_file, "wb") as f:
            f.write(compressor.encode(pkl))
        icf = vcf2zarr.IntermediateColumnarFormat(icf_path)
        with pytest.raises(ValueError, match="Corruption detected"):
            icf["POS"].values  # noqa B018
        with pytest.raises(ValueError, match="Corruption detected"):
            list(icf["POS"].iter_values(0, 9))


class TestSlicing:
    data_path = "tests/data/vcf/multi_contig.vcf.gz"

    @pytest.fixture(scope="class")
    def icf(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("data") / "example.exploded"
        return vcf2zarr.explode(
            out, [self.data_path], column_chunk_size=0.0125, worker_processes=0
        )

    def test_repr(self, icf):
        assert repr(icf).startswith(
            "IntermediateColumnarFormat(fields=7, partitions=5, records=4665, path="
        )

    def test_pos_repr(self, icf):
        assert repr(icf["POS"]).startswith(
            "IntermediateColumnarFormatField(name=POS, "
            "partition_chunks=[8, 8, 8, 8, 8], path="
        )

    def test_partition_record_index(self, icf):
        nt.assert_array_equal(
            icf.partition_record_index, [0, 933, 1866, 2799, 3732, 4665]
        )

    def test_pos_values(self, icf):
        field = icf["POS"]
        pos = np.array([v[0] for v in field.values])
        # Check the actual values here to make sure other tests make sense
        actual = np.hstack([1 + np.arange(933) for _ in range(5)])
        nt.assert_array_equal(pos, actual)

    def test_pos_chunk_records(self, icf):
        pos = icf["POS"]
        for j in range(pos.num_partitions):
            a = pos.chunk_record_index(j)
            nt.assert_array_equal(a, [0, 118, 236, 354, 472, 590, 708, 826, 933])
            a = pos.chunk_num_records(j)
            nt.assert_array_equal(a, [118, 118, 118, 118, 118, 118, 118, 107])
            assert len(a) == pos.num_chunks(j)

    @pytest.mark.parametrize(
        ("start", "stop"),
        [
            (0, 1),
            (0, 4665),
            (100, 200),
            (100, 500),
            (100, 1000),
            (100, 1500),
            (100, 4500),
            (2000, 2500),
            (118, 237),
            (710, 850),
            (931, 1000),
            (1865, 1867),
            (1866, 2791),
            (2732, 3200),
            (2798, 2799),
            (2799, 2800),
            (4664, 4665),
        ],
    )
    def test_slice(self, icf, start, stop):
        field = icf["POS"]
        pos = np.array(field.values)
        pos_slice = np.array(list(field.iter_values(start, stop)))
        nt.assert_array_equal(pos[start:stop], pos_slice)


@pytest.mark.parametrize(
    "regions",
    [
        # Overlapping partitions
        [("1", 100, 200), ("1", 150, 250)],
        # Overlap by one position
        [("1", 100, 201), ("1", 200, 300)],
        # End coord is *inclusive*
        [("1", 100, 201), ("1", 201, 300)],
        # Contained overlap
        [("1", 100, 300), ("1", 150, 250)],
        # Exactly equal
        [("1", 100, 200), ("1", 100, 200)],
    ],
)
def test_check_overlap(regions):
    partitions = [
        icf_mod.VcfPartition("", region=vcf_utils.Region(contig, start, end))
        for contig, start, end in regions
    ]
    with pytest.raises(ValueError, match="Overlapping VCF regions"):
        icf_mod.check_overlapping_partitions(partitions)
