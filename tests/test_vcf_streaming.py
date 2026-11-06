import json
import sys
from unittest import mock

import click.testing
import numpy as np
import numpy.testing as nt
import pytest

from bio2zarr import cli
from bio2zarr import vcf as vcf_mod

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="VCF support requires cyvcf2"
)


def assert_zarr_equal(left, right):
    assert left.attrs.asdict() == right.attrs.asdict()
    assert set(left.array_keys()) == set(right.array_keys())
    for name in left.array_keys():
        a, b = left[name], right[name]
        assert a.attrs.asdict() == b.attrs.asdict(), name
        assert a.dtype == b.dtype, name
        assert a.shape == b.shape, name
        assert a.chunks == b.chunks, name
        x, y = a[:], b[:]
        if a.dtype.kind == "f":
            x = x.view(f"u{a.dtype.itemsize}")
            y = y.view(f"u{b.dtype.itemsize}")
        nt.assert_array_equal(x, y, err_msg=name)


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize(
    "filename",
    [
        "sample.vcf.gz",
        "sample.bcf",
        "mixed_ploidy.vcf.gz",
        "triploid.vcf.gz",
        "field_type_combos.vcf.gz",
        "sample_no_genotypes.vcf.gz",
        "sample_no_genotypes_with_gt_header.vcf.gz",
        "sample_all_missing_gts.vcf.gz",
        "empty.vcf",
        "empty.vcf.gz",
    ],
)
def test_direct_and_batched_match_icf(tmp_path, filename, zarr_format):
    path = f"tests/data/vcf/{filename}"
    icf = vcf_mod.explode(tmp_path / "icf", [path])
    expected = vcf_mod.encode(
        icf.path,
        variants_chunk_size=3,
        samples_chunk_size=2,
        zarr_format=zarr_format,
    )
    batched = vcf_mod.encode(
        icf.path,
        variants_chunk_size=3,
        samples_chunk_size=2,
        batch_size=2,
        zarr_format=zarr_format,
    )
    assert_zarr_equal(expected, batched)
    with mock.patch.object(vcf_mod, "explode", side_effect=AssertionError("ICF used")):
        actual = vcf_mod.convert(
            [path],
            infer_shape=True,
            variants_chunk_size=3,
            samples_chunk_size=2,
            batch_size=2,
            zarr_format=zarr_format,
        )
    assert_zarr_equal(expected, actual)


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("bound", [2, 4])
def test_explicit_max_ploidy(tmp_path, workers, bound):
    path = "tests/data/vcf/sample.vcf.gz"
    icf = vcf_mod.explode(tmp_path / "icf", [path])
    expected = vcf_mod.encode(icf.path, ploidy=bound, variants_chunk_size=3)
    with mock.patch.object(vcf_mod, "explode", side_effect=AssertionError("ICF used")):
        actual = vcf_mod.convert(
            [path],
            max_ploidy=bound,
            worker_processes=workers,
            variants_chunk_size=3,
            batch_size=2,
        )
    assert_zarr_equal(expected, actual)


def test_reusable_dimensions():
    path = "tests/data/vcf/mixed_ploidy.vcf.gz"
    dimensions = vcf_mod.scan_vcf_dimensions([path])
    with mock.patch.object(vcf_mod, "scan_vcf_dimensions", side_effect=AssertionError):
        result = vcf_mod.convert(
            [path], infer_shape=True, dimension_metadata=dimensions
        )
    assert result["call_genotype"].shape[2] == 3


@pytest.mark.parametrize("bound", [0, -1, True, 1.5])
def test_invalid_ploidy(bound):
    with pytest.raises(ValueError, match="max_ploidy must be"):
        vcf_mod.convert(["tests/data/vcf/sample.vcf.gz"], max_ploidy=bound)


def test_ploidy_overflow():
    with pytest.raises(ValueError, match="exceeds max (ploidy|genotypes)"):
        vcf_mod.convert(["tests/data/vcf/triploid.vcf.gz"], max_ploidy=2)


def test_gt_only_ploidy_overflow(tmp_path):
    path = tmp_path / "input.vcf"
    path.write_text(
        "##fileformat=VCFv4.3\n##contig=<ID=1>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts\n"
        "1\t1\t.\tA\tC\t.\t.\t.\tGT\t0/0/1\n"
    )
    with pytest.raises(ValueError, match="ploidy 3 exceeds max ploidy 2"):
        vcf_mod.convert([path], max_ploidy=2)


@pytest.mark.parametrize("option", ["batch_size", "max_batch_bytes"])
def test_invalid_batch_option(option):
    with pytest.raises(ValueError, match=option):
        vcf_mod.convert(["tests/data/vcf/sample.vcf.gz"], **{option: 0})


def test_oversized_row():
    with pytest.raises(ValueError, match="exceeds max_batch_bytes"):
        vcf_mod.convert(
            ["tests/data/vcf/sample.vcf.gz"], max_ploidy=2, max_batch_bytes=100
        )


def test_batch_payload_and_chunk_bounds(tmp_path):
    path = "tests/data/vcf/sample.vcf.gz"
    dimensions = vcf_mod.scan_vcf_dimensions([path])
    metadata, header = dimensions
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"metadata": metadata.asdict(), "header": header}))
    source = vcf_mod.DirectVcfSource(manifest)
    schema = source.generate_schema(variants_chunk_size=3)
    converter = vcf_mod.VcfRowConverter(source, schema)
    sizes = [
        sum(vcf_mod.value_nbytes(v) for v in converter.convert(row).values())
        for row in source.iter_rows(0, 9)
    ]
    budget = max(sizes) + 1
    writes = []
    original = vcf_mod.BatchedVcfZarrWriter.write_batch

    def record(arrays, start, batch):
        payload = sum(
            vcf_mod.value_nbytes(value) for row in batch for value in row.values()
        )
        writes.append((start, len(batch), payload))
        original(arrays, start, batch)

    with mock.patch.object(
        vcf_mod.BatchedVcfZarrWriter, "write_batch", side_effect=record
    ):
        result = vcf_mod.convert(
            [path],
            infer_shape=True,
            dimension_metadata=dimensions,
            variants_chunk_size=3,
            batch_size=64,
            max_batch_bytes=budget,
        )
    assert result["variant_position"].shape == (9,)
    assert sum(length for _, length, _ in writes) == 9
    assert len(writes) > 3  # byte budget splits batches before chunk boundaries
    for start, length, payload in writes:
        assert payload <= budget
        assert 1 <= length <= 64
        assert start // 3 == (start + length - 1) // 3


@pytest.mark.parametrize("batch_size", [None, 2])
def test_default_batch_matches_output_chunk(batch_size):
    writes = []
    original = vcf_mod.BatchedVcfZarrWriter.write_batch

    def record(arrays, start, batch):
        writes.append((start, len(batch)))
        original(arrays, start, batch)

    with mock.patch.object(
        vcf_mod.BatchedVcfZarrWriter, "write_batch", side_effect=record
    ):
        vcf_mod.convert(
            ["tests/data/vcf/sample.vcf.gz"],
            infer_shape=True,
            variants_chunk_size=4,
            batch_size=batch_size,
        )
    expected = [(0, 4), (4, 4), (8, 1)]
    if batch_size == 2:
        expected = [(0, 2), (2, 2), (4, 2), (6, 2), (8, 1)]
    assert writes == expected


def test_default_batch_exceeds_legacy_row_cap(tmp_path):
    path = tmp_path / "input.vcf"
    path.write_text(
        "##fileformat=VCFv4.3\n##contig=<ID=1>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        + "".join(f"1\t{j + 1}\t.\tA\tC\t.\t.\t.\n" for j in range(1005))
    )
    writes = []
    original = vcf_mod.BatchedVcfZarrWriter.write_batch

    def record(arrays, start, batch):
        writes.append((start, len(batch)))
        original(arrays, start, batch)

    with mock.patch.object(
        vcf_mod.BatchedVcfZarrWriter, "write_batch", side_effect=record
    ):
        result = vcf_mod.convert([path], infer_shape=True, variants_chunk_size=1004)
    assert writes == [(0, 1004), (1004, 1)]
    nt.assert_array_equal(result["variant_position"][:], np.arange(1, 1006))


def test_integer_dtype_bound(tmp_path):
    icf = vcf_mod.explode(tmp_path / "icf", ["tests/data/vcf/sample.vcf.gz"])
    schema = icf.generate_schema()
    schema.field_map()["variant_position"].dtype = "i1"
    file = tmp_path / "schema.json"
    file.write_text(schema.asjson())
    with pytest.raises(ValueError, match="POS values exceed schema dtype"):
        vcf_mod.encode(icf.path, schema_path=file, batch_size=2)


def test_implicit_batch_for_byte_budget(tmp_path):
    icf = vcf_mod.explode(tmp_path / "icf", ["tests/data/vcf/sample.vcf.gz"])
    with pytest.raises(ValueError, match="row exceeds max_batch_bytes"):
        vcf_mod.encode(icf.path, max_batch_bytes=100)


def test_distributed_batched(tmp_path):
    icf = vcf_mod.explode(tmp_path / "icf", ["tests/data/vcf/sample.vcf.gz"])
    expected = vcf_mod.encode(icf.path, variants_chunk_size=3)
    out = tmp_path / "out"
    summary = vcf_mod.encode_init(icf.path, out, 3, variants_chunk_size=3, batch_size=2)
    for partition in range(summary.num_partitions):
        vcf_mod.encode_partition(out, partition)
    vcf_mod.encode_finalise(out)
    vcf_mod.create_index(out)
    assert_zarr_equal(expected, vcf_mod.zarr_utils.open_vcf_zarr(out))


def test_custom_float_schema(tmp_path):
    icf = vcf_mod.explode(tmp_path / "icf", ["tests/data/vcf/sample.vcf.gz"])
    for dtype in ["f2", "f4", "f8"]:
        schema = icf.generate_schema()
        for field in schema.fields:
            if np.dtype(field.dtype).kind == "f":
                field.dtype = dtype
        file = tmp_path / "schema.json"
        file.write_text(schema.asjson())
        expected = vcf_mod.encode(icf.path, schema_path=file)
        actual = vcf_mod.encode(icf.path, schema_path=file, batch_size=2)
        assert_zarr_equal(expected, actual)


def test_field_overflow_rejected(tmp_path):
    icf = vcf_mod.explode(tmp_path / "icf", ["tests/data/vcf/sample.vcf.gz"])
    schema = icf.generate_schema()
    schema.field_map()["call_HQ"].dimensions = ("variants", "samples")
    file = tmp_path / "schema.json"
    file.write_text(schema.asjson())
    with pytest.raises(ValueError, match="FORMAT/HQ value shape .* exceeds schema"):
        vcf_mod.encode(icf.path, schema_path=file, batch_size=2)


def test_local_alleles(tmp_path):
    path = tmp_path / "local.vcf"
    path.write_text(
        "##fileformat=VCFv4.3\n##contig=<ID=1>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Depth">\n'
        '##FORMAT=<ID=PL,Number=G,Type=Integer,Description="Likelihood">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\n"
        "1\t1\t.\tA\tC,G\t.\t.\t.\tGT:AD:PL\t0/1:1,2,3:0,1,2,3,4,5"
        "\t1|2:4,5,6:6,5,4,3,2,1\n"
        "1\t2\t.\tA\tC\t.\t.\t.\tGT\t0/1\t./.\n"
        "1\t3\t.\tA\tC\t.\t.\t.\tGT:AD:PL\t0/0:1,2:0,1,2\t1/1:3,4:4,3,2\n"
    )
    icf = vcf_mod.explode(tmp_path / "icf", [path])
    expected = vcf_mod.encode(icf.path, local_alleles=True)
    actual = vcf_mod.convert([path], max_ploidy=2, local_alleles=True, batch_size=2)
    assert_zarr_equal(expected, actual)


def test_cli_direct(tmp_path):
    out = tmp_path / "out"
    result = click.testing.CliRunner().invoke(
        cli.vcf2zarr_main,
        [
            "convert",
            "tests/data/vcf/sample.vcf.gz",
            str(out),
            "--max-ploidy",
            "2",
            "--batch-size",
            "2",
            "-Q",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert vcf_mod.zarr_utils.open_vcf_zarr(out)["call_genotype"].shape == (9, 3, 2)


def test_cli_conflicting_options(tmp_path):
    result = click.testing.CliRunner().invoke(
        cli.vcf2zarr_main,
        [
            "convert",
            "tests/data/vcf/sample.vcf.gz",
            str(tmp_path / "out"),
            "--max-ploidy",
            "2",
            "--ploidy",
            "2",
        ],
    )
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output


@pytest.mark.parametrize("bound", [0, -1, True, 2.5])
def test_invalid_max_alleles(bound):
    with pytest.raises(ValueError, match="max_alleles must be a positive integer"):
        vcf_mod.convert(["tests/data/vcf/sample.vcf.gz"], max_alleles=bound)
    with pytest.raises(ValueError, match="max_alleles must be a positive integer"):
        vcf_mod.scan_vcf_dimensions(["tests/data/vcf/sample.vcf.gz"], max_alleles=bound)


@pytest.mark.parametrize("bound", [1, 2, 3])
def test_max_alleles_overflow(bound):
    with pytest.raises(ValueError, match=f"exceeds max_alleles={bound}"):
        vcf_mod.convert(["tests/data/vcf/sample.vcf.gz"], max_alleles=bound)


@pytest.mark.parametrize("max_ploidy", [None, 2, 4])
@pytest.mark.parametrize("infer_shape", [False, True])
def test_allele_bound_and_shape_inference(tmp_path, max_ploidy, infer_shape):
    path = "tests/data/vcf/sample.vcf.gz"
    icf = vcf_mod.explode(tmp_path / "icf", [path])
    schema = icf.generate_schema(ploidy=max_ploidy, max_alleles=5)
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(schema.asjson())
    expected = vcf_mod.encode(icf.path, schema_path=schema_path)
    with mock.patch.object(vcf_mod, "explode", side_effect=AssertionError("ICF used")):
        actual = vcf_mod.convert(
            [path], max_alleles=5, max_ploidy=max_ploidy, infer_shape=infer_shape
        )
    assert_zarr_equal(expected, actual)
    assert actual["variant_allele"].shape == (9, 5)
    assert actual["call_genotype"].shape[2] == (max_ploidy or 2)


def test_max_alleles_number_g_padding(tmp_path):
    path = "tests/data/vcf/triploid.vcf.gz"
    icf = vcf_mod.explode(tmp_path / "icf", [path])
    schema = icf.generate_schema(max_alleles=4)
    file = tmp_path / "schema.json"
    file.write_text(schema.asjson())
    expected = vcf_mod.encode(icf.path, schema_path=file)
    actual = vcf_mod.convert([path], max_alleles=4, infer_shape=True)
    assert_zarr_equal(expected, actual)
    assert actual["call_PL"].shape[2] == 20


def test_allele_bound_with_reused_dimensions():
    path = "tests/data/vcf/sample.vcf.gz"
    dimensions = vcf_mod.scan_vcf_dimensions([path])
    with pytest.raises(ValueError, match="exceeds max_alleles=2"):
        vcf_mod.convert([path], max_alleles=2, dimension_metadata=dimensions)
    actual = vcf_mod.convert([path], max_alleles=5, dimension_metadata=dimensions)
    assert actual["variant_allele"].shape == (9, 5)


def test_bounded_scan_reused_with_bound():
    path = "tests/data/vcf/sample.vcf.gz"
    dimensions = vcf_mod.scan_vcf_dimensions([path], max_alleles=5, max_ploidy=2)
    root = vcf_mod.convert(
        [path],
        infer_shape=True,
        max_alleles=5,
        max_ploidy=2,
        dimension_metadata=dimensions,
    )
    assert root["variant_allele"].shape == (9, 5)


def test_genotype_allele_bound(tmp_path):
    path = tmp_path / "input.vcf"
    path.write_text(
        "##fileformat=VCFv4.3\n##contig=<ID=1>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts\n"
        "1\t1\t.\tA\tC\t.\t.\t.\tGT\t0/2\n"
    )
    with pytest.raises(ValueError, match="Genotype allele index exceeds allele-count"):
        vcf_mod.convert([path], max_alleles=2)


def test_reused_metadata_genotype_dtype_not_truncated(tmp_path):
    path = tmp_path / "input.vcf"
    path.write_text(
        "##fileformat=VCFv4.3\n##contig=<ID=1>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts\n"
        f"1\t1\t.\tA\t{','.join(['C'] * 128)}\t.\t.\t.\tGT\t0/128\n"
    )
    dimensions = vcf_mod.scan_vcf_dimensions([path], max_alleles=129)
    gt = next(f for f in dimensions[0].fields if f.name == "GT")
    gt.summary.max_value = 1  # stale/narrow metadata must not permit an i1 wrap
    with pytest.raises(ValueError, match="Genotype allele index exceeds schema dtype"):
        vcf_mod.convert([path], max_alleles=129, dimension_metadata=dimensions)


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_max_alleles_one_monomorphic(tmp_path, zarr_format):
    path = tmp_path / "monomorphic.vcf"
    path.write_text(
        "##fileformat=VCFv4.3\n##contig=<ID=1>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '##INFO=<ID=AC,Number=A,Type=Integer,Description="Count">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts\n"
        "1\t1\t.\tA\t.\t.\t.\t.\tGT\t0/0\n"
    )
    actual = vcf_mod.convert([path], max_alleles=1, zarr_format=zarr_format)
    assert actual["variant_allele"].shape == (1, 1)
    assert actual["variant_AC"].shape == (1, 0)
    nt.assert_array_equal(actual["call_genotype"][:], [[[0, 0]]])


def test_no_gt_number_g_bounds(tmp_path):
    path = tmp_path / "input.vcf"
    path.write_text(
        "##fileformat=VCFv4.3\n##contig=<ID=1>\n"
        '##INFO=<ID=GP,Number=G,Type=Float,Description="Probability">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "1\t1\t.\tA\tC\t.\t.\tGP=0.2,0.3,0.5\n"
    )
    root = vcf_mod.convert([path], max_alleles=3, max_ploidy=2)
    assert root["variant_GP"].shape == (1, 6)
    nt.assert_allclose(root["variant_GP"][0, :3], [0.2, 0.3, 0.5])
    with pytest.raises(ValueError, match="exceeds max genotypes"):
        vcf_mod.convert([path], max_alleles=2, max_ploidy=1)


def test_max_alleles_with_variable_fields():
    path = "tests/data/vcf/field_type_combos.vcf.gz"
    dimensions = vcf_mod.scan_vcf_dimensions([path])
    observed = next(
        f.summary.max_number + 1 for f in dimensions[0].fields if f.name == "ALT"
    )
    root = vcf_mod.convert([path], max_alleles=max(2, observed))
    expected = vcf_mod.convert([path], infer_shape=True)
    assert_zarr_equal(expected, root)


def test_shape_with_known_ploidy_without_allele_bound():
    root = vcf_mod.convert(
        ["tests/data/vcf/sample.vcf.gz"], infer_shape=True, max_ploidy=2
    )
    assert root["call_genotype"].shape == (9, 3, 2)
    assert root["variant_allele"].shape == (9, 4)


def test_legacy_ploidy_padding_with_shape_inference():
    root = vcf_mod.convert(["tests/data/vcf/sample.vcf.gz"], infer_shape=True, ploidy=4)
    assert root["call_genotype"].shape == (9, 3, 4)
    nt.assert_array_equal(root["call_genotype"][:, :, 2:], -2)


def test_cli_shape_and_bounds(tmp_path):
    out = tmp_path / "out"
    result = click.testing.CliRunner().invoke(
        cli.vcf2zarr_main,
        [
            "convert",
            "tests/data/vcf/sample.vcf.gz",
            str(out),
            "--infer-shape",
            "--max-ploidy",
            "2",
            "--max-alleles",
            "5",
            "-Q",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert vcf_mod.zarr_utils.open_vcf_zarr(out)["variant_allele"].shape == (9, 5)


def test_nonlocal_output_rejected_before_overwrite():
    result = click.testing.CliRunner().invoke(
        cli.vcf2zarr_main,
        [
            "convert",
            "tests/data/vcf/sample.vcf.gz",
            "remote://bucket/existing",
            "--max-ploidy",
            "2",
            "--force",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 2
    assert "Final VCZ output must be a local path" in result.output
    with pytest.raises(ValueError, match="Final VCZ output must be a local path"):
        vcf_mod.convert(
            ["tests/data/vcf/sample.vcf.gz"], "remote://bucket/existing", max_ploidy=2
        )
