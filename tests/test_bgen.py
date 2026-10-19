"""Small synthetic BGEN fixtures, encoded independently of the reader.

The layout 2 encoding follows the BGEN 1.2/1.3 specification:
https://www.chg.ox.ac.uk/~gav/bgen_format/spec/latest.html
No external or user data is used.
"""

import importlib.metadata
import math
import re
import struct
import subprocess
import sys
from unittest import mock

import cbgen
import click.testing
import numpy as np
import numpy.testing as nt
import pytest
import zarr
import zstandard

from bio2zarr import bgen, cli, vcf
from bio2zarr.vcz_verification import assert_all_missing_float
from tests.utils import load_dataset


def encode_string(value, width=2):
    value = value.encode("utf-8")
    return len(value).to_bytes(width, "little") + value


def write_bgen(
    path,
    *,
    nvariants=7,
    compression=1,
    layout=2,
    bits=8,
    embedded=True,
    phased=False,
    ploidy=(2, 2, 2, 2),
    alleles=("GG", "T"),
    position=10,
    position_step=1,
    chrom="1",
    invalid=False,
):
    """Write four samples with known integer probabilities and missingness."""
    nsamples = len(ploidy)
    denominator = 2**bits - 1
    # Include deterministic, uncertain, and missing calls.
    q = np.array(
        [
            [denominator, 0],
            [0, denominator],
            [denominator // 5, denominator // 3],
            [0, 0],
        ],
        dtype=np.uint64,
    )
    if invalid:
        q[0] = denominator
    missing = np.array([False, False, False, True])
    expected = np.column_stack((q, denominator - q.sum(axis=1))).astype(float)
    expected /= denominator
    expected[missing] = np.nan
    sample_block = b""
    if embedded:
        names = b"".join(encode_string(f"person_{j}") for j in range(nsamples))
        sample_block = struct.pack("<II", 8 + len(names), nsamples) + names
    flags = compression | (layout << 2) | (int(embedded) << 31)
    header = struct.pack(
        "<IIII4sI", 20 + len(sample_block), 20, nvariants, nsamples, b"bgen", flags
    )
    with open(path, "wb") as f:
        f.write(header + sample_block)
        for j in range(nvariants):
            f.write(encode_string(f"snp_{j}"))
            f.write(encode_string(f"rs{j}" if j != 1 else ""))
            f.write(encode_string(chrom if j < nvariants // 2 else "X"))
            f.write(struct.pack("<IH", position + j * position_step, len(alleles)))
            for allele in alleles:
                f.write(encode_string(allele, width=4))
            flags = [p | (128 if m or j == 2 else 0) for p, m in zip(ploidy, missing)]
            block = struct.pack(
                "<IHBB", nsamples, len(alleles), min(ploidy), max(ploidy)
            )
            block += bytes(flags) + bytes([phased, bits])
            if phased or any(p != 2 for p in ploidy) or len(alleles) != 2:
                values = []
                for p in ploidy:
                    count = (
                        p * (len(alleles) - 1)
                        if phased
                        else math.comb(p + len(alleles) - 1, len(alleles) - 1) - 1
                    )
                    values.extend([0] * count)
            else:
                values = q.ravel().tolist()
            packed = sum(int(v) << (bits * k) for k, v in enumerate(values))
            block += packed.to_bytes((bits * len(values) + 7) // 8, "little")
            if compression == 0:
                data = block
            elif compression == 1:
                import zlib  # noqa: PLC0415

                data = struct.pack("<I", len(block)) + zlib.compress(block)
            else:
                data = struct.pack("<I", len(block)) + zstandard.compress(block)
            f.write(struct.pack("<I", len(data)) + data)
    return expected


@pytest.fixture
def example(tmp_path):
    path = tmp_path / "example.bgen"
    expected = write_bgen(path)
    return path, expected


@pytest.mark.parametrize("compression", [1, 2])
@pytest.mark.parametrize("bits", [1, 8, 14, 23, 32])
@pytest.mark.parametrize("zarr_format", [2, 3])
def test_probabilities(example, compression, bits, zarr_format):
    path, _ = example
    expected = write_bgen(path, compression=compression, bits=bits)
    root = bgen.convert(
        path, variants_chunk_size=3, samples_chunk_size=2, zarr_format=zarr_format
    )
    assert root.metadata.zarr_format == zarr_format
    assert root["call_GP"].chunks == (3, 2, 3)
    assert root["call_DS"].chunks == (3, 2)
    assert root["call_GP"].dtype == np.dtype("f8")
    for j in range(7):
        if j == 2:
            assert_all_missing_float(root["call_GP"][j])
            assert_all_missing_float(root["call_DS"][j])
        else:
            nt.assert_allclose(root["call_GP"][j], expected, atol=0, rtol=1e-15)
            nt.assert_allclose(
                root["call_DS"][j],
                expected[:, 1] + 2 * expected[:, 2],
                atol=0,
                rtol=1e-15,
            )
            assert_all_missing_float(root["call_GP"][j, -1])
            assert_all_missing_float(root["call_DS"][j, -1])
    assert "call_genotype" not in root


def test_metadata(example):
    path, _ = example
    root = bgen.convert(path)
    nt.assert_array_equal(root["sample_id"][:], [f"person_{j}" for j in range(4)])
    nt.assert_array_equal(root["contig_id"][:], ["1", "X"])
    nt.assert_array_equal(root["variant_contig"][:], [0, 0, 0, 1, 1, 1, 1])
    nt.assert_array_equal(root["variant_position"][:], np.arange(10, 17))
    nt.assert_array_equal(root["variant_allele"][:], [["GG", "T"]] * 7)
    nt.assert_array_equal(root["variant_length"][:], [2] * 7)
    nt.assert_array_equal(root["variant_bgen_id"][:], [f"snp_{j}" for j in range(7)])
    nt.assert_array_equal(
        root["variant_id"][:], ["rs0", ".", "rs2", "rs3", "rs4", "rs5", "rs6"]
    )
    nt.assert_array_equal(root["variant_id_mask"][:], [False, True] + [False] * 5)
    assert root.attrs["vcf_zarr_version"] == "0.5"
    ds = load_dataset(root)
    assert ds["call_GP"].dims == ("variants", "samples", "genotypes")
    assert ds["call_DS"].dims == ("variants", "samples")
    assert list(path.parent.glob("*.metafile")) == []


@pytest.mark.parametrize("embedded", [False, True])
def test_sample_ids(example, embedded):
    path, _ = example
    write_bgen(path, embedded=embedded)
    root = bgen.convert(path)
    prefix = "person" if embedded else "sample"
    nt.assert_array_equal(root["sample_id"][:], [f"{prefix}_{j}" for j in range(4)])
    root = bgen.convert(path, sample_ids=["a", "b", "c", "d"])
    nt.assert_array_equal(root["sample_id"][:], ["a", "b", "c", "d"])


@pytest.mark.parametrize(
    "ids",
    [
        [],
        ["one"],
        ["a"] * 4,
        ["", "a", "b", "c"],
        [".", "a", "b", "c"],
        [1, 2, 3, 4],
        "abcd",
    ],
)
def test_bad_sample_ids(example, ids, tmp_path):
    path, _ = example
    out = tmp_path / "output.vcz"
    with pytest.raises(ValueError, match="sample_ids"):
        bgen.convert(path, out, sample_ids=ids)
    assert not out.exists()


@pytest.mark.parametrize("option", ["variants_chunk_size", "samples_chunk_size"])
@pytest.mark.parametrize("value", [0, -1, 1.5])
def test_bad_chunk_size(example, option, value):
    with pytest.raises(ValueError, match=f"{option} must be a positive integer"):
        bgen.convert(example[0], **{option: value})


@pytest.mark.parametrize("worker_processes", [0, 1, 2])
def test_index_and_chunk_boundaries(example, worker_processes, tmp_path):
    path, expected = example
    write_bgen(path, nvariants=263)
    root = bgen.convert(
        path,
        tmp_path / "output.vcz",
        variants_chunk_size=17,
        samples_chunk_size=3,
        worker_processes=worker_processes,
    )
    assert root["call_GP"].shape == (263, 4, 3)
    for j in [0, 16, 17, 127, 128, 131, 132, 255, 262]:
        nt.assert_allclose(root["call_GP"][j], expected, atol=0, rtol=1e-15)


@pytest.mark.parametrize("out_type", ["memory", "directory", "zip"])
@pytest.mark.parametrize("mode", ["r", "r+"])
def test_outputs(example, tmp_path, out_type, mode):
    path, _ = example
    out = (
        None
        if out_type == "memory"
        else tmp_path / ("output.vcz.zip" if out_type == "zip" else "output.vcz")
    )
    if out_type == "zip" and mode != "r":
        with pytest.raises(ValueError, match="Zip output is read-only"):
            bgen.convert(path, out, mode=mode)
        assert not out.exists()
        return
    root = bgen.convert(path, out, mode=mode)
    assert root.read_only == (mode == "r")
    assert root["call_GP"].shape == (7, 4, 3)
    if out_type == "memory":
        assert isinstance(root.store, zarr.storage.MemoryStore)
    elif out_type == "zip":
        assert isinstance(root.store, zarr.storage.ZipStore)
        root.store.close()


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"compression": 3}, "(?i)compression|flags"),
        ({"layout": 3}, "(?i)layout|version|flags"),
        ({"layout": 0}, "(?i)layout|version|flags"),
        ({"nvariants": 0}, "at least one variant"),
    ],
)
def test_invalid_input_before_output(example, kwargs, error, tmp_path, capfd):
    path, _ = example
    write_bgen(path, **kwargs)
    out = tmp_path / "output.vcz"
    with pytest.raises((ValueError, RuntimeError)) as exc_info:
        bgen.convert(path, out)
    assert re.search(error, str(exc_info.value) + capfd.readouterr().err)
    assert not out.exists()


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"position": 0}, "position must be positive"),
        ({"chrom": ""}, "missing chromosome"),
        ({"alleles": ("", "C")}, "missing allele"),
        ({"invalid": True}, "invalid genotype probabilities"),
    ],
)
def test_unsupported_data_preflight(example, kwargs, error, tmp_path):
    path, _ = example
    write_bgen(path, **kwargs)
    out = tmp_path / "output.vcz"
    with pytest.raises(ValueError, match=error):
        bgen.convert(path, out)
    assert not out.exists()


def test_large_position(example):
    path, _ = example
    write_bgen(path, position=2**31)
    root = bgen.convert(path)
    assert root["variant_position"].dtype == np.dtype("i8")
    nt.assert_array_equal(root["variant_position"][:], np.arange(2**31, 2**31 + 7))


def test_large_region_record_count_with_small_positions(example):
    path, _ = example
    write_bgen(path, nvariants=300, position_step=0)
    root = bgen.convert(path, variants_chunk_size=300)
    assert root["variant_position"].dtype == np.dtype("i4")
    nt.assert_array_equal(root["region_index"][:, 5], [150, 150])


def test_reference_span_crosses_int32_boundary(example):
    path, _ = example
    write_bgen(path, position=2**31 - 1, position_step=0)
    root = bgen.convert(path)
    assert root["variant_position"].dtype == np.dtype("i8")
    nt.assert_array_equal(root["region_index"][:, 4], [2**31, 2**31])


def test_unsorted_positions(example, tmp_path):
    path, _ = example
    write_bgen(path, position=100, position_step=-1)
    out = tmp_path / "output.vcz"
    with pytest.raises(ValueError, match="positions must be nondecreasing"):
        bgen.convert(path, out)
    assert not out.exists()


@pytest.mark.parametrize("contents", [b"", b"not bgen at all" * 2])
def test_invalid_header(tmp_path, contents):
    path = tmp_path / "bad.bgen"
    path.write_bytes(contents)
    with pytest.raises((ValueError, RuntimeError), match="(?i)bgen|header"):
        bgen.convert(path)


def test_dependency_attribution(example):
    assert importlib.metadata.version("bgen-limix").startswith("4.")
    real_distribution = importlib.metadata.distribution

    def no_bgen(name):
        if name == "bgen-limix":
            raise importlib.metadata.PackageNotFoundError(name)
        return real_distribution(name)

    with mock.patch("importlib.metadata.distribution", side_effect=no_bgen):
        with pytest.raises(ImportError, match="not the legacy cbgen"):
            bgen.convert(example[0])


def test_missing_dependency(example):
    with mock.patch.dict(sys.modules, {"cbgen": None}):
        with pytest.raises(ImportError, match=r"bio2zarr\[bgen\]"):
            bgen.convert(example[0])


def test_unsupported_backend_api(example):
    class OldReader:
        pass

    with mock.patch.object(cbgen, "BgenFile", OldReader):
        with pytest.raises(ImportError, match=r"bgen-limix>=4\.13\.0"):
            bgen.convert(example[0])


def test_backend_check_and_reader_factory_used(example):
    readers = []
    original = bgen.BgenFormat.open_reader

    def open_reader(source):
        reader = original(source)
        readers.append(reader)
        return reader

    with (
        mock.patch.object(bgen, "check_backend", wraps=bgen.check_backend) as check,
        mock.patch.object(bgen.BgenFormat, "open_reader", open_reader),
        mock.patch.object(
            cbgen.BgenFile,
            "create_metafile",
            autospec=True,
            side_effect=cbgen.BgenFile.create_metafile,
        ) as create,
    ):
        bgen.convert(example[0])
    check.assert_called_once_with()
    assert create.call_args.args[0] is readers[0]


def test_conflicting_legacy_distribution(example):
    original = importlib.metadata.distribution

    def with_legacy(name):
        if name == "cbgen":
            return mock.sentinel.legacy
        return original(name)

    with mock.patch("importlib.metadata.distribution", side_effect=with_legacy):
        with pytest.raises(ImportError, match="legacy cbgen distribution conflicts"):
            bgen.convert(example[0])


@pytest.mark.parametrize("mode", ["w", "a", "bad"])
def test_invalid_mode(example, mode):
    with pytest.raises(ValueError, match="mode must be"):
        bgen.convert(example[0], mode=mode)


def test_bounded_reads(example):
    path, _ = example
    original = cbgen.BgenFile.read_genotypes_batch
    shapes = []
    offsets = []

    def read(source, batch_offsets, *args, **kwargs):
        assert kwargs["max_ncomb"] == 3
        assert kwargs["precision"] == 64
        offsets.extend(batch_offsets)
        genotype = original(source, batch_offsets, *args, **kwargs)
        shapes.append(genotype.probability.shape)
        return genotype

    with (
        mock.patch.object(cbgen.BgenFile, "read_genotypes_batch", read),
        mock.patch.object(cbgen.BgenFile, "read_genotype", side_effect=AssertionError),
    ):
        bgen.convert(path, variants_chunk_size=2)
    assert shapes
    assert shapes == [(2, 4, 3)] * 3 + [(1, 4, 3)]
    assert len(offsets) == len(set(offsets)) == 7


def test_cli(example, tmp_path):
    path, expected = example
    out = tmp_path / "output.vcz"
    ids = tmp_path / "ids.txt"
    ids.write_text("a\nb\nc\nd\n")
    result = click.testing.CliRunner().invoke(
        cli.bgen2zarr_main,
        [
            "convert",
            str(path),
            str(out),
            "--sample-ids",
            str(ids),
            "-l",
            "3",
            "-w",
            "2",
            "--zarr-format",
            "3",
            "-Q",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    root = zarr.open_group(out)
    assert root.metadata.zarr_format == 3
    nt.assert_array_equal(root["sample_id"][:], ["a", "b", "c", "d"])
    nt.assert_allclose(root["call_GP"][0], expected)
    result = subprocess.run(
        [sys.executable, "-m", "bio2zarr", "bgen2zarr", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "convert" in result.stdout
    zip_path = tmp_path / "cli.vcz.zip"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "bio2zarr",
            "bgen2zarr",
            "convert",
            str(path),
            str(zip_path),
            "--zarr-format",
            "2",
            "-Q",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    with zarr.storage.ZipStore(zip_path, mode="r") as store:
        root = zarr.open_group(store, mode="r")
        nt.assert_allclose(root["call_GP"][0], expected)


@pytest.mark.skipif(sys.platform == "win32", reason="VCF support requires cyvcf2")
def test_against_vcf_converter(example, tmp_path):
    import pysam  # noqa: PLC0415

    path, expected = example
    vcf_path = tmp_path / "expected.vcf"
    lines = [
        "##fileformat=VCFv4.3",
        "##contig=<ID=1>",
        "##contig=<ID=X>",
        '##FORMAT=<ID=GP,Number=G,Type=Float,Description="Probabilities">',
        '##FORMAT=<ID=DS,Number=1,Type=Float,Description="Alt dosage">',
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        + "\t".join(f"person_{j}" for j in range(4)),
    ]
    for j in range(7):
        calls = []
        for p in expected:
            if j == 2 or np.isnan(p).any():
                calls.append(".,.,.:.")
            else:
                calls.append(",".join(str(v) for v in p) + ":" + str(p[1] + 2 * p[2]))
        lines.append(
            f"{'1' if j < 3 else 'X'}\t{10 + j}\t"
            f"{'rs' + str(j) if j != 1 else '.'}\tGG\tT\t.\t.\t.\tGP:DS\t"
            + "\t".join(calls)
        )
    vcf_path.write_text("\n".join(lines) + "\n")
    compressed_path = str(vcf_path) + ".gz"
    pysam.tabix_compress(str(vcf_path), compressed_path)
    pysam.tabix_index(compressed_path, preset="vcf")
    bgen_root = bgen.convert(path)
    vcf_root = vcf.convert([compressed_path])
    for name in (
        "variant_position",
        "variant_contig",
        "variant_allele",
        "variant_length",
        "variant_id",
        "variant_id_mask",
        "sample_id",
        "contig_id",
    ):
        nt.assert_array_equal(bgen_root[name][:], vcf_root[name][:])
    for name in ("call_GP", "call_DS"):
        nt.assert_allclose(bgen_root[name][:], vcf_root[name][:], rtol=1e-6)
