import json

import pytest
import sgkit as sg
import xarray.testing as xt
import zarr

from bio2zarr import core, vcf, vcf_utils


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
        ("arg", "expected"),
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

    @pytest.mark.parametrize("max_memory", ["150KiB", "200KiB"])
    def test_not_enough_memory_for_two(
        self, tmp_path, icf_path, zarr_path, caplog, max_memory
    ):
        other_zarr_path = tmp_path / "zarr"
        with caplog.at_level("WARNING"):
            vcf.encode(
                icf_path, other_zarr_path, max_memory=max_memory, worker_processes=2
            )
        assert "Limiting number of workers to 1 to keep within" in caplog.text
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
        with open(icf_path / "metadata.json") as f:
            d = json.load(f)

        d["format_version"] = version
        with pytest.raises(
            ValueError, match="Intermediate columnar metadata format version mismatch"
        ):
            vcf.IcfMetadata.fromdict(d)

    @pytest.mark.parametrize("version", ["0.0", "1.0", "xxxxx", 0.1])
    def test_encode_metadata_mismatch(self, tmpdir, icf_path, version):
        zarr_path = tmpdir / "zarr"
        vcf.encode_init(icf_path, zarr_path, 1)
        with open(zarr_path / "wip" / "metadata.json") as f:
            d = json.load(f)
        d["format_version"] = version
        with pytest.raises(ValueError, match="VcfZarrWriter format version mismatch"):
            vcf.VcfZarrWriterMetadata.fromdict(d)


class TestEncodeDimensionSeparator:
    @pytest.mark.parametrize("dimension_separator", [None, "/"])
    def test_directories(self, tmp_path, icf_path, dimension_separator):
        zarr_path = tmp_path / "zarr"
        vcf.encode(icf_path, zarr_path, dimension_separator=dimension_separator)
        # print(zarr_path)
        chunk_file = zarr_path / "call_genotype" / "0" / "0" / "0"
        assert chunk_file.exists()

    def test_files(self, tmp_path, icf_path):
        zarr_path = tmp_path / "zarr"
        vcf.encode(icf_path, zarr_path, dimension_separator=".")
        chunk_file = zarr_path / "call_genotype" / "0.0.0"
        assert chunk_file.exists()

    @pytest.mark.parametrize("dimension_separator", ["\\", "X", []])
    def test_bad_value(self, tmp_path, icf_path, dimension_separator):
        zarr_path = tmp_path / "zarr"
        with pytest.raises(ValueError, match="dimension_separator must be either"):
            vcf.encode(icf_path, zarr_path, dimension_separator=dimension_separator)


class TestSchemaJsonRoundTrip:
    def assert_json_round_trip(self, schema):
        schema2 = vcf.VcfZarrSchema.fromjson(schema.asjson())
        assert schema == schema2

    def test_generated_no_changes(self, icf_path):
        icf = vcf.IntermediateColumnarFormat(icf_path)
        self.assert_json_round_trip(vcf.VcfZarrSchema.generate(icf))

    def test_generated_no_columns(self, icf_path):
        icf = vcf.IntermediateColumnarFormat(icf_path)
        schema = vcf.VcfZarrSchema.generate(icf)
        schema.columns.clear()
        self.assert_json_round_trip(schema)

    def test_generated_no_samples(self, icf_path):
        icf = vcf.IntermediateColumnarFormat(icf_path)
        schema = vcf.VcfZarrSchema.generate(icf)
        schema.sample_id.clear()
        self.assert_json_round_trip(schema)

    def test_generated_change_dtype(self, icf_path):
        icf = vcf.IntermediateColumnarFormat(icf_path)
        schema = vcf.VcfZarrSchema.generate(icf)
        schema.columns["variant_position"].dtype = "i8"
        self.assert_json_round_trip(schema)

    def test_generated_change_compressor(self, icf_path):
        icf = vcf.IntermediateColumnarFormat(icf_path)
        schema = vcf.VcfZarrSchema.generate(icf)
        schema.columns["variant_position"].compressor = {"cname": "FAKE"}
        self.assert_json_round_trip(schema)


class TestSchemaEncode:
    @pytest.mark.parametrize(
        ("cname", "clevel", "shuffle"), [("lz4", 1, 0), ("zlib", 7, 1), ("zstd", 4, 2)]
    )
    def test_codec(self, tmp_path, icf_path, cname, clevel, shuffle):
        zarr_path = tmp_path / "zarr"
        icf = vcf.IntermediateColumnarFormat(icf_path)
        schema = vcf.VcfZarrSchema.generate(icf)
        for var in schema.columns.values():
            var.compressor["cname"] = cname
            var.compressor["clevel"] = clevel
            var.compressor["shuffle"] = shuffle
        schema_path = tmp_path / "schema"
        with open(schema_path, "w") as f:
            f.write(schema.asjson())
        vcf.encode(icf_path, zarr_path, schema_path=schema_path)
        root = zarr.open(zarr_path)
        for var in schema.columns.values():
            a = root[var.name]
            assert a.compressor.cname == cname
            assert a.compressor.clevel == clevel
            assert a.compressor.shuffle == shuffle

    @pytest.mark.parametrize("dtype", ["i4", "i8"])
    def test_genotype_dtype(self, tmp_path, icf_path, dtype):
        zarr_path = tmp_path / "zarr"
        icf = vcf.IntermediateColumnarFormat(icf_path)
        schema = vcf.VcfZarrSchema.generate(icf)
        schema.columns["call_genotype"].dtype = dtype
        schema_path = tmp_path / "schema"
        with open(schema_path, "w") as f:
            f.write(schema.asjson())
        vcf.encode(icf_path, zarr_path, schema_path=schema_path)
        root = zarr.open(zarr_path)
        assert root["call_genotype"].dtype == dtype


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
            "dtype": "i1",
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

    def test_call_GQ(self, schema):
        assert schema["columns"]["call_GQ"] == {
            "name": "call_GQ",
            "dtype": "i1",
            "shape": [9, 3],
            "chunks": [10000, 1000],
            "dimensions": ["variants", "samples"],
            "description": "Genotype Quality",
            "vcf_field": "FORMAT/GQ",
            "compressor": {
                "id": "blosc",
                "cname": "zstd",
                "clevel": 7,
                "shuffle": 0,
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
        vcf.VcfPartition("", region=vcf_utils.Region(contig, start, end))
        for contig, start, end in regions
    ]
    with pytest.raises(ValueError, match="Overlapping VCF regions"):
        vcf.check_overlapping_partitions(partitions)


class TestVcfDescriptions:
    @pytest.mark.parametrize(
        ("field", "description"),
        [
            ("variant_NS", "Number of Samples With Data"),
            ("variant_AN", "Total number of alleles in called genotypes"),
            (
                "variant_AC",
                "Allele count in genotypes, for each ALT allele, "
                "in the same order as listed",
            ),
            ("variant_DP", "Total Depth"),
            ("variant_AF", "Allele Frequency"),
            ("variant_AA", "Ancestral Allele"),
            ("variant_DB", "dbSNP membership, build 129"),
            ("variant_H2", "HapMap2 membership"),
            ("call_GQ", "Genotype Quality"),
            ("call_DP", "Read Depth"),
            ("call_HQ", "Haplotype Quality"),
        ],
    )
    def test_fields(self, schema, field, description):
        assert schema["columns"][field]["description"] == description

    # This information is not in the schema yet,
    # https://github.com/sgkit-dev/bio2zarr/issues/123
    # @pytest.mark.parametrize(
    #     ("filt", "description"),
    #     [
    #         ("s50","Less than 50% of samples have data"),
    #         ("q10", "Quality below 10"),
    #     ])
    # def test_filters(self, schema, filt, description):
    #     assert schema["filters"][field]["description"] == description


class TestVcfZarrWriterExample:
    arrays = (
        "variant_contig",
        "variant_filter",
        "variant_id",
        "variant_AA",
        "variant_AC",
        "variant_AF",
        "variant_AN",
        "variant_DB",
        "variant_DP",
        "variant_H2",
        "variant_NS",
        "variant_position",
        "variant_quality",
        "variant_allele",
        "call_DP",
        "call_GQ",
        "call_genotype",
        "call_genotype_phased",
        "call_genotype_mask",
        "call_HQ",
    )

    def test_init_paths(self, icf_path, tmp_path):
        zarr_path = tmp_path / "x.zarr"
        assert not zarr_path.exists()
        num_partitions, _ = vcf.encode_init(
            icf_path, zarr_path, 7, variants_chunk_size=3
        )
        assert num_partitions == 3
        assert zarr_path.exists()
        wip_path = zarr_path / "wip"
        assert wip_path.exists()
        wip_partitions_path = wip_path / "partitions"
        assert wip_partitions_path.exists()
        wip_arrays_path = wip_path / "arrays"
        assert wip_arrays_path.exists()
        for name in self.arrays:
            array_path = wip_arrays_path / name
            assert array_path.exists()
        with open(wip_path / "metadata.json") as f:
            d = json.loads(f.read())
            # Basic test
            assert len(d["partitions"]) == 3

    def test_finalise_paths(self, icf_path, tmp_path):
        zarr_path = tmp_path / "x.zarr"
        assert not zarr_path.exists()
        num_partitions, _ = vcf.encode_init(
            icf_path, zarr_path, 7, variants_chunk_size=3
        )
        wip_path = zarr_path / "wip"
        assert wip_path.exists()
        for j in range(num_partitions):
            vcf.encode_partition(zarr_path, j)
            assert (wip_path / "partitions" / f"p{j}").exists()
        vcf.encode_finalise(zarr_path)
        assert zarr_path.exists()
        assert not wip_path.exists()

    def test_finalise_no_partitions_fails(self, icf_path, tmp_path):
        zarr_path = tmp_path / "x.zarr"
        vcf.encode_init(icf_path, zarr_path, 3, variants_chunk_size=3)
        with pytest.raises(
            FileNotFoundError, match="Partitions not encoded: \\[0, 1, 2\\]"
        ):
            vcf.encode_finalise(zarr_path)

    @pytest.mark.parametrize("partition", [0, 1, 2])
    def test_finalise_missing_partition_fails(self, icf_path, tmp_path, partition):
        zarr_path = tmp_path / "x.zarr"
        vcf.encode_init(icf_path, zarr_path, 3, variants_chunk_size=3)
        for j in range(3):
            if j != partition:
                vcf.encode_partition(zarr_path, j)
        with pytest.raises(
            FileNotFoundError, match=f"Partitions not encoded: \\[{partition}\\]"
        ):
            vcf.encode_finalise(zarr_path)

    @pytest.mark.parametrize("partition", [0, 1, 2])
    def test_encode_partition(self, icf_path, tmp_path, partition):
        zarr_path = tmp_path / "x.zarr"
        vcf.encode_init(icf_path, zarr_path, 3, variants_chunk_size=3)
        partition_path = zarr_path / "wip" / "partitions" / f"p{partition}"
        assert not partition_path.exists()
        vcf.encode_partition(zarr_path, partition)
        assert partition_path.exists()

    def test_double_encode_partition(self, icf_path, tmp_path, caplog):
        partition = 1
        zarr_path = tmp_path / "x.zarr"
        vcf.encode_init(icf_path, zarr_path, 3, variants_chunk_size=3)
        partition_path = zarr_path / "wip" / "partitions" / f"p{partition}"
        assert not partition_path.exists()
        vcf.encode_partition(zarr_path, partition)
        assert partition_path.exists()
        size = core.du(partition_path)
        assert size > 0
        with caplog.at_level("WARNING"):
            vcf.encode_partition(zarr_path, partition)
        assert "Removing existing partition at" in caplog.text
        assert partition_path.exists()
        assert core.du(partition_path) == size

    @pytest.mark.parametrize("partition", [-1, 3, 100])
    def test_encode_partition_out_of_range(self, icf_path, tmp_path, partition):
        zarr_path = tmp_path / "x.zarr"
        vcf.encode_init(icf_path, zarr_path, 3, variants_chunk_size=3)
        with pytest.raises(ValueError, match="Partition index must be in the range"):
            vcf.encode_partition(zarr_path, partition)
