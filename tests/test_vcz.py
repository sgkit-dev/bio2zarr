import json

import pysam
import pytest
import sgkit as sg
import xarray.testing as xt
import zarr

from bio2zarr import core, vcf2zarr
from bio2zarr.vcf2zarr import icf as icf_mod
from bio2zarr.vcf2zarr import vcz as vcz_mod
from bio2zarr.zarr_utils import zarr_v3


@pytest.fixture(scope="module")
def vcf_file():
    return "tests/data/vcf/sample.vcf.gz"


@pytest.fixture(scope="module")
def icf_path(vcf_file, tmp_path_factory):
    out = tmp_path_factory.mktemp("data") / "example.exploded"
    vcf2zarr.explode(out, [vcf_file])
    return out


@pytest.fixture(scope="module")
def schema_path(icf_path, tmp_path_factory):
    out = tmp_path_factory.mktemp("data") / "example.schema.json"
    with open(out, "w") as f:
        vcf2zarr.mkschema(icf_path, f)
    return out


@pytest.fixture(scope="module")
def schema(schema_path):
    with open(schema_path) as f:
        return vcf2zarr.VcfZarrSchema.fromjson(f.read())


@pytest.fixture(scope="module")
def zarr_path(icf_path, tmp_path_factory):
    out = tmp_path_factory.mktemp("data") / "example.zarr"
    vcf2zarr.encode(icf_path, out)
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
        assert vcz_mod.parse_max_memory(arg) == expected

    @pytest.mark.parametrize("max_memory", [-1, 0, 1, "100 bytes"])
    def test_not_enough_memory(self, tmp_path, icf_path, max_memory):
        zarr_path = tmp_path / "zarr"
        with pytest.raises(ValueError, match="Insufficient memory"):
            vcf2zarr.encode(icf_path, zarr_path, max_memory=max_memory)

    @pytest.mark.parametrize("max_memory", ["315KiB", "500KiB"])
    def test_not_enough_memory_for_two(
        self, tmp_path, icf_path, zarr_path, caplog, max_memory
    ):
        other_zarr_path = tmp_path / "zarr"
        with caplog.at_level("WARNING"):
            vcf2zarr.encode(
                icf_path, other_zarr_path, max_memory=max_memory, worker_processes=2
            )
        assert "Limiting number of workers to 1 to keep within" in caplog.text
        ds1 = sg.load_dataset(zarr_path)
        ds2 = sg.load_dataset(other_zarr_path)
        xt.assert_equal(ds1, ds2)


class TestJsonVersions:
    @pytest.mark.parametrize("version", ["0.1", "1.0", "xxxxx", 0.2])
    def test_zarr_schema_mismatch(self, schema, version):
        d = schema.asdict()
        d["format_version"] = version
        with pytest.raises(ValueError, match="Zarr schema format version mismatch"):
            vcf2zarr.VcfZarrSchema.fromdict(d)

    @pytest.mark.parametrize("version", ["0.0", "1.0", "xxxxx", 0.1])
    def test_exploded_metadata_mismatch(self, tmpdir, icf_path, version):
        with open(icf_path / "metadata.json") as f:
            d = json.load(f)

        d["format_version"] = version
        with pytest.raises(
            ValueError, match="Intermediate columnar metadata format version mismatch"
        ):
            icf_mod.IcfMetadata.fromdict(d)

    @pytest.mark.parametrize("version", ["0.0", "1.0", "xxxxx", 0.1])
    def test_encode_metadata_mismatch(self, tmpdir, icf_path, version):
        zarr_path = tmpdir / "zarr"
        vcf2zarr.encode_init(icf_path, zarr_path, 1)
        with open(zarr_path / "wip" / "metadata.json") as f:
            d = json.load(f)
        d["format_version"] = version
        with pytest.raises(ValueError, match="VcfZarrWriter format version mismatch"):
            vcz_mod.VcfZarrWriterMetadata.fromdict(d)


@pytest.mark.skipif(
    zarr_v3(), reason="Zarr-python v3 does not support dimension_separator"
)
class TestEncodeDimensionSeparator:
    @pytest.mark.parametrize("dimension_separator", [None, "/"])
    def test_directories(self, tmp_path, icf_path, dimension_separator):
        zarr_path = tmp_path / "zarr"
        vcf2zarr.encode(icf_path, zarr_path, dimension_separator=dimension_separator)
        # print(zarr_path)
        chunk_file = zarr_path / "call_genotype" / "0" / "0" / "0"
        assert chunk_file.exists()

    def test_files(self, tmp_path, icf_path):
        zarr_path = tmp_path / "zarr"
        vcf2zarr.encode(icf_path, zarr_path, dimension_separator=".")
        chunk_file = zarr_path / "call_genotype" / "0.0.0"
        assert chunk_file.exists()

    @pytest.mark.parametrize("dimension_separator", ["\\", "X", []])
    def test_bad_value(self, tmp_path, icf_path, dimension_separator):
        zarr_path = tmp_path / "zarr"
        with pytest.raises(ValueError, match="dimension_separator must be either"):
            vcf2zarr.encode(
                icf_path, zarr_path, dimension_separator=dimension_separator
            )


class TestSchemaChunkSize:
    @pytest.mark.parametrize(
        ("samples_chunk_size", "variants_chunk_size"),
        [
            (1, 2),
            (2, 1),
            (3, 5),
        ],
    )
    def test_chunk_sizes(self, icf_path, samples_chunk_size, variants_chunk_size):
        icf = vcf2zarr.IntermediateColumnarFormat(icf_path)
        schema = vcf2zarr.VcfZarrSchema.generate(
            icf,
            variants_chunk_size=variants_chunk_size,
            samples_chunk_size=samples_chunk_size,
        )
        assert schema.samples_chunk_size == samples_chunk_size
        assert schema.variants_chunk_size == variants_chunk_size
        found = 0
        for field in schema.fields:
            assert field.dimensions[0] == "variants"
            assert field.chunks[0] == variants_chunk_size
            if "samples" in field.dimensions:
                dim = field.dimensions.index("samples")
                assert field.chunks[dim] == samples_chunk_size
                found += 1
        assert found > 0


class TestSchemaJsonRoundTrip:
    def assert_json_round_trip(self, schema):
        schema2 = vcf2zarr.VcfZarrSchema.fromjson(schema.asjson())
        assert schema == schema2

    def test_generated_no_changes(self, icf_path):
        icf = vcf2zarr.IntermediateColumnarFormat(icf_path)
        self.assert_json_round_trip(vcf2zarr.VcfZarrSchema.generate(icf))

    def test_generated_no_fields(self, icf_path):
        icf = vcf2zarr.IntermediateColumnarFormat(icf_path)
        schema = vcf2zarr.VcfZarrSchema.generate(icf)
        schema.fields.clear()
        self.assert_json_round_trip(schema)

    def test_generated_no_samples(self, icf_path):
        icf = vcf2zarr.IntermediateColumnarFormat(icf_path)
        schema = vcf2zarr.VcfZarrSchema.generate(icf)
        schema.samples.clear()
        self.assert_json_round_trip(schema)

    def test_generated_change_dtype(self, icf_path):
        icf = vcf2zarr.IntermediateColumnarFormat(icf_path)
        schema = vcf2zarr.VcfZarrSchema.generate(icf)
        schema.field_map()["variant_position"].dtype = "i8"
        self.assert_json_round_trip(schema)

    def test_generated_change_compressor(self, icf_path):
        icf = vcf2zarr.IntermediateColumnarFormat(icf_path)
        schema = vcf2zarr.VcfZarrSchema.generate(icf)
        schema.field_map()["variant_position"].compressor = {"cname": "FAKE"}
        self.assert_json_round_trip(schema)


class TestSchemaEncode:
    @pytest.mark.parametrize(
        ("cname", "clevel", "shuffle"), [("lz4", 1, 0), ("zlib", 7, 1), ("zstd", 4, 2)]
    )
    def test_codec(self, tmp_path, icf_path, cname, clevel, shuffle):
        zarr_path = tmp_path / "zarr"
        icf = vcf2zarr.IntermediateColumnarFormat(icf_path)
        schema = vcf2zarr.VcfZarrSchema.generate(icf)
        for array_spec in schema.fields:
            array_spec.compressor["cname"] = cname
            array_spec.compressor["clevel"] = clevel
            array_spec.compressor["shuffle"] = shuffle
        schema_path = tmp_path / "schema"
        with open(schema_path, "w") as f:
            f.write(schema.asjson())
        vcf2zarr.encode(icf_path, zarr_path, schema_path=schema_path)
        root = zarr.open(zarr_path)
        for array_spec in schema.fields:
            a = root[array_spec.name]
            assert a.compressor.cname == cname
            assert a.compressor.clevel == clevel
            assert a.compressor.shuffle == shuffle

    @pytest.mark.parametrize("dtype", ["i4", "i8"])
    def test_genotype_dtype(self, tmp_path, icf_path, dtype):
        zarr_path = tmp_path / "zarr"
        icf = vcf2zarr.IntermediateColumnarFormat(icf_path)
        schema = vcf2zarr.VcfZarrSchema.generate(icf)
        schema.field_map()["call_genotype"].dtype = dtype
        schema_path = tmp_path / "schema"
        with open(schema_path, "w") as f:
            f.write(schema.asjson())
        vcf2zarr.encode(icf_path, zarr_path, schema_path=schema_path)
        root = zarr.open(zarr_path)
        assert root["call_genotype"].dtype == dtype


def get_field_dict(a_schema, name):
    d = a_schema.asdict()
    for field in d["fields"]:
        if field["name"] == name:
            return field


class TestChunkNbytes:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("call_genotype", 54),  # 9 * 3 * 2 * 1
            ("call_genotype_phased", 27),
            ("call_genotype_mask", 54),
            ("variant_position", 36),  # 9 * 4
            ("variant_H2", 9),
            ("variant_AC", 18),  # 9 * 2
            # Object fields have an itemsize of 8
            ("variant_AA", 72),  # 9 * 8
            ("variant_allele", 9 * 4 * 8),
        ],
    )
    def test_example_schema(self, schema, field, value):
        field = schema.field_map()[field]
        assert field.chunk_nbytes == value

    def test_chunk_size(self, icf_path, tmp_path):
        store = vcf2zarr.IntermediateColumnarFormat(icf_path)
        schema = vcf2zarr.VcfZarrSchema.generate(
            store, samples_chunk_size=2, variants_chunk_size=3
        )
        fields = schema.field_map()
        assert fields["call_genotype"].chunk_nbytes == 3 * 2 * 2
        assert fields["variant_position"].chunk_nbytes == 3 * 4
        assert fields["variant_AC"].chunk_nbytes == 3 * 2


class TestValidateSchema:
    @pytest.mark.parametrize("size", [2**31, 2**31 + 1, 2**32])
    def test_chunk_too_large(self, schema, size):
        schema = vcf2zarr.VcfZarrSchema.fromdict(schema.asdict())
        field = schema.field_map()["variant_H2"]
        field.shape = (size,)
        field.chunks = (size,)
        with pytest.raises(ValueError, match="Field variant_H2 chunks are too large"):
            schema.validate()

    @pytest.mark.parametrize("size", [2**31 - 1, 2**30])
    def test_chunk_not_too_large(self, schema, size):
        schema = vcf2zarr.VcfZarrSchema.fromdict(schema.asdict())
        field = schema.field_map()["variant_H2"]
        field.shape = (size,)
        field.chunks = (size,)
        schema.validate()


class TestDefaultSchema:
    def test_format_version(self, schema):
        assert schema.format_version == vcz_mod.ZARR_SCHEMA_FORMAT_VERSION

    def test_chunk_size(self, schema):
        assert schema.samples_chunk_size == 1000
        assert schema.variants_chunk_size == 10000

    def test_samples(self, schema):
        assert schema.asdict()["samples"] == [
            {"id": s} for s in ["NA00001", "NA00002", "NA00003"]
        ]

    def test_contigs(self, schema):
        assert schema.asdict()["contigs"] == [
            {"id": s, "length": None} for s in ["19", "20", "X"]
        ]

    def test_filters(self, schema):
        assert schema.asdict()["filters"] == [
            {"id": "PASS", "description": "All filters passed"},
            {"id": "s50", "description": "Less than 50% of samples have data"},
            {"id": "q10", "description": "Quality below 10"},
        ]

    def test_variant_contig(self, schema):
        assert get_field_dict(schema, "variant_contig") == {
            "name": "variant_contig",
            "dtype": "i1",
            "shape": (9,),
            "chunks": (10000,),
            "dimensions": ("variants",),
            "description": "An identifier from the reference genome or an "
            "angle-bracketed ID string pointing to a contig in the assembly file",
            "vcf_field": None,
            "compressor": {
                "id": "blosc",
                "cname": "zstd",
                "clevel": 7,
                "shuffle": 0,
                "blocksize": 0,
            },
            "filters": tuple(),
        }

    def test_call_genotype(self, schema):
        assert get_field_dict(schema, "call_genotype") == {
            "name": "call_genotype",
            "dtype": "i1",
            "shape": (9, 3, 2),
            "chunks": (10000, 1000, 2),
            "dimensions": ("variants", "samples", "ploidy"),
            "description": "",
            "vcf_field": None,
            "compressor": {
                "id": "blosc",
                "cname": "zstd",
                "clevel": 7,
                "shuffle": 2,
                "blocksize": 0,
            },
            "filters": tuple(),
        }

    def test_call_genotype_mask(self, schema):
        assert get_field_dict(schema, "call_genotype_mask") == {
            "name": "call_genotype_mask",
            "dtype": "bool",
            "shape": (9, 3, 2),
            "chunks": (10000, 1000, 2),
            "dimensions": ("variants", "samples", "ploidy"),
            "description": "",
            "vcf_field": None,
            "compressor": {
                "id": "blosc",
                "cname": "zstd",
                "clevel": 7,
                "shuffle": 2,
                "blocksize": 0,
            },
            "filters": tuple(),
        }

    def test_call_genotype_phased(self, schema):
        assert get_field_dict(schema, "call_genotype_mask") == {
            "name": "call_genotype_mask",
            "dtype": "bool",
            "shape": (9, 3, 2),
            "chunks": (10000, 1000, 2),
            "dimensions": ("variants", "samples", "ploidy"),
            "description": "",
            "vcf_field": None,
            "compressor": {
                "id": "blosc",
                "cname": "zstd",
                "clevel": 7,
                "shuffle": 2,
                "blocksize": 0,
            },
            "filters": tuple(),
        }

    def test_call_GQ(self, schema):
        assert get_field_dict(schema, "call_GQ") == {
            "name": "call_GQ",
            "dtype": "i1",
            "shape": (9, 3),
            "chunks": (10000, 1000),
            "dimensions": ("variants", "samples"),
            "description": "Genotype Quality",
            "vcf_field": "FORMAT/GQ",
            "compressor": {
                "id": "blosc",
                "cname": "zstd",
                "clevel": 7,
                "shuffle": 0,
                "blocksize": 0,
            },
            "filters": tuple(),
        }


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
        assert schema.field_map()[field].description == description

    @pytest.mark.parametrize(
        ("filt", "description"),
        [
            ("PASS", "All filters passed"),
            ("s50", "Less than 50% of samples have data"),
            ("q10", "Quality below 10"),
        ],
    )
    def test_filters(self, schema, filt, description):
        d = {f.id: f.description for f in schema.filters}
        assert d[filt] == description


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
        summary = vcf2zarr.encode_init(icf_path, zarr_path, 7, variants_chunk_size=3)
        assert summary.num_partitions == 3
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
        summary = vcf2zarr.encode_init(icf_path, zarr_path, 7, variants_chunk_size=3)
        wip_path = zarr_path / "wip"
        assert wip_path.exists()
        for j in range(summary.num_partitions):
            vcf2zarr.encode_partition(zarr_path, j)
            assert (wip_path / "partitions" / f"p{j}").exists()
        vcf2zarr.encode_finalise(zarr_path)
        assert zarr_path.exists()
        assert not wip_path.exists()

    def test_finalise_no_partitions_fails(self, icf_path, tmp_path):
        zarr_path = tmp_path / "x.zarr"
        vcf2zarr.encode_init(icf_path, zarr_path, 3, variants_chunk_size=3)
        with pytest.raises(
            FileNotFoundError, match="Partitions not encoded: \\[0, 1, 2\\]"
        ):
            vcf2zarr.encode_finalise(zarr_path)

    @pytest.mark.parametrize("partition", [0, 1, 2])
    def test_finalise_missing_partition_fails(self, icf_path, tmp_path, partition):
        zarr_path = tmp_path / "x.zarr"
        vcf2zarr.encode_init(icf_path, zarr_path, 3, variants_chunk_size=3)
        for j in range(3):
            if j != partition:
                vcf2zarr.encode_partition(zarr_path, j)
        with pytest.raises(
            FileNotFoundError, match=f"Partitions not encoded: \\[{partition}\\]"
        ):
            vcf2zarr.encode_finalise(zarr_path)

    @pytest.mark.parametrize("partition", [0, 1, 2])
    def test_encode_partition(self, icf_path, tmp_path, partition):
        zarr_path = tmp_path / "x.zarr"
        vcf2zarr.encode_init(icf_path, zarr_path, 3, variants_chunk_size=3)
        partition_path = zarr_path / "wip" / "partitions" / f"p{partition}"
        assert not partition_path.exists()
        vcf2zarr.encode_partition(zarr_path, partition)
        assert partition_path.exists()

    def test_double_encode_partition(self, icf_path, tmp_path, caplog):
        partition = 1
        zarr_path = tmp_path / "x.zarr"
        vcf2zarr.encode_init(icf_path, zarr_path, 3, variants_chunk_size=3)
        partition_path = zarr_path / "wip" / "partitions" / f"p{partition}"
        assert not partition_path.exists()
        vcf2zarr.encode_partition(zarr_path, partition)
        assert partition_path.exists()
        size = core.du(partition_path)
        assert size > 0
        with caplog.at_level("WARNING"):
            vcf2zarr.encode_partition(zarr_path, partition)
        assert "Removing existing partition at" in caplog.text
        assert partition_path.exists()
        assert core.du(partition_path) == size

    @pytest.mark.parametrize("partition", [-1, 3, 100])
    def test_encode_partition_out_of_range(self, icf_path, tmp_path, partition):
        zarr_path = tmp_path / "x.zarr"
        vcf2zarr.encode_init(icf_path, zarr_path, 3, variants_chunk_size=3)
        with pytest.raises(ValueError, match="Partition index not in the valid range"):
            vcf2zarr.encode_partition(zarr_path, partition)


class TestClobberFixedFields:
    def generate_vcf(self, path, info_field=None, format_field=None, num_rows=1):
        with open(path, "w") as out:
            print("##fileformat=VCFv4.2", file=out)
            print('##FILTER=<ID=PASS,Description="All filters passed">', file=out)
            print("##contig=<ID=1>", file=out)
            if info_field is not None:
                print(
                    f'##INFO=<ID={info_field},Number=1,Type=Float,Description="">',
                    file=out,
                )
            if format_field is not None:
                print(
                    f'##FORMAT=<ID={format_field},Number=1,Type=Float,Description="">',
                    file=out,
                )
            header = "\t".join(
                ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"]
            )
            print(header, file=out)
            for k in range(num_rows):
                pos = str(k + 1)
                print("\t".join(["1", pos, "A", "T", ".", ".", ".", "."]), file=out)

        # print(open(path).read())
        # This also compresses the input file
        pysam.tabix_index(str(path), preset="vcf")

    @pytest.mark.parametrize(
        "field",
        [
            "contig",
            "id",
            "id_mask",
            "position",
            "allele",
            "filter",
            "quality",
        ],
    )
    def test_variant_fields(self, tmp_path, field):
        vcf_file = tmp_path / "test.vcf"
        self.generate_vcf(vcf_file, info_field=field)
        with pytest.raises(ValueError, match=f"INFO field name.*{field}"):
            vcf2zarr.explode(tmp_path / "x.icf", [tmp_path / "test.vcf.gz"])

    @pytest.mark.parametrize(
        "field",
        [
            "genotype",
            "genotype_phased",
            "genotype_mask",
        ],
    )
    def test_call_fields(self, tmp_path, field):
        vcf_file = tmp_path / "test.vcf"
        self.generate_vcf(vcf_file, format_field=field)
        with pytest.raises(ValueError, match=f"FORMAT field name.*{field}"):
            vcf2zarr.explode(tmp_path / "x.icf", [tmp_path / "test.vcf.gz"])


class TestBadSchemaChanges:
    # [{'id': 'NA00001'}, {'id': 'NA00002'}, {'id': 'NA00003'}],
    @pytest.mark.parametrize(
        "samples",
        [
            [],
            [{"id": "NA00001"}, {"id": "NA00003"}],
            [{"id": "NA00001"}, {"id": "NA00002"}, {"id": "NA00004"}],
            [
                {"id": "NA00001"},
                {"id": "NA00002"},
                {"id": "NA00003"},
                {"id": "NA00004"},
            ],
            [{"id": "NA00001"}, {"id": "NA00003"}, {"id": "NA00002"}],
        ],
    )
    def test_removed_samples(self, tmp_path, schema, icf_path, samples):
        d = schema.asdict()
        d["samples"] = samples
        schema_path = tmp_path / "schema.json"
        with open(schema_path, "w") as f:
            json.dump(d, f)
        with pytest.raises(ValueError, match="Subsetting or reordering samples"):
            vcf2zarr.encode(icf_path, tmp_path / "z", schema_path=schema_path)
