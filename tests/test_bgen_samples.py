"""Oxford/QCTOOL sample identifiers, including the vcztools export profile."""

import pathlib
from unittest import mock

import cbgen
import click.testing
import numpy.testing as nt
import pytest
import zarr

from bio2zarr import bgen, cli
from tests.test_bgen import write_bgen
from tests.test_bgen_reference import example_prefix


@pytest.fixture
def sample_file(tmp_path):
    path = tmp_path / "example.sample"
    path.write_text("ID_1 ID_2 missing\n0 0 0\nz z 0\na a 0\nb b 0\nc c 0\n")
    return path


def test_reference_sample_file():
    path = pathlib.Path("tests/data/bgen/example.sample")
    assert bgen.read_sample_file(path) == [f"sample_{j:03}" for j in range(1, 501)]


def test_reference_sample_conversion(tmp_path):
    root = bgen.convert(
        example_prefix(tmp_path),
        sample_file=pathlib.Path("tests/data/bgen/example.sample"),
        variants_chunk_size=41,
        samples_chunk_size=113,
    )
    nt.assert_array_equal(
        root["sample_id"][:], [f"sample_{j:03}" for j in range(1, 501)]
    )
    assert root["call_GP"].shape == (98, 500, 3)


@pytest.mark.parametrize(
    ("contents", "column", "expected"),
    [
        ("ID_1\n0\nz\na\n", None, ["z", "a"]),
        ("ID\n0\n001\n0\n", None, ["001", "0"]),
        (
            "ID sex case covariate\n0 D B C\nz M control 0.1\na F case NA\n",
            None,
            ["z", "a"],
        ),
        ("ID_1 ID_2 missing\n0 0 0\nfamily z 0\nfamily a NA\n", "ID_2", ["z", "a"]),
        (
            "ID_1\tID_2\tmissing\tphenotype\n0 0 0 P\nfam1 z 0 NA\nfam2 a 0 3\n",
            None,
            ["fam1", "fam2"],
        ),
        ("ID_1 ID_2 missing\n0 0 0\nfam1 z 0\nfam2 a 0\n", "ID_1", ["fam1", "fam2"]),
    ],
)
def test_parser(tmp_path, contents, column, expected):
    path = tmp_path / "input.sample"
    path.write_text(contents)
    assert bgen.read_sample_file(path, sample_id_column=column) == expected


@pytest.mark.parametrize(
    ("contents", "column", "line", "message"),
    [
        ("", None, 1, "nonempty column header"),
        ("\n0\nz\n", None, 1, "nonempty column header"),
        ("ID_1 ID_1\n0 0\nz z\n", None, 1, "column names must be unique"),
        ("ID_1\n", None, 2, "expected 1 column types, found 0"),
        ("ID_1 ID_2\n0\nz z\n", None, 2, "expected 2 column types"),
        ("ID_1\nD\nz\n", None, 2, "first identifier column must have type 0"),
        ("ID_1 pheno\n0 X\nz 1\n", None, 2, "invalid type 'X'"),
        ("ID_1 pheno\n0 0\nz 1\n", None, 2, "type 0 is not allowed"),
        ("ID_1\n0\nz\n", "ID_2", 1, "is not in the header"),
        ("ID_1 trait\n0 D\nz a\n", "trait", 2, "must have type 0"),
        ("ID_1 missing\n0 0\nz 0\n", "missing", 2, "must have type 0"),
        ("ID_1\n0\n", None, 3, "at least one sample row"),
        ("ID_1\n0\n\n", None, 3, "expected 1 columns, found 0"),
        ("ID_1 ID_2\n0 0\nz\n", None, 3, "expected 2 columns, found 1"),
        ("ID_1\n0\nz extra\n", None, 3, "expected 1 columns, found 2"),
        ("ID_1\n0\nNA\n", None, 3, "missing identifier"),
        ("ID_1\n0\n.\n", None, 3, "missing identifier"),
        ("ID_1\n0\nz\nz\n", None, 4, "duplicate identifier"),
        (
            "ID_1 ID_2\n0 0\nfamily z\nfamily a\n",
            None,
            4,
            "sample_id_column / --sample-id-column",
        ),
        ("ID_1 ID_2\n0 0\nfam1 a\nfam2 a\n", "ID_2", 4, "duplicate identifier"),
        ("ID_1 ID_2\n0 0\nfam1 NA\n", "ID_2", 3, "missing identifier"),
    ],
)
def test_invalid_parser(tmp_path, contents, column, line, message):
    path = tmp_path / "invalid.sample"
    path.write_text(contents)
    with pytest.raises(ValueError, match="Oxford sample file") as exc:
        bgen.read_sample_file(path, sample_id_column=column)
    assert f"{path}, line {line}:" in str(exc.value)
    assert message in str(exc.value)


@pytest.mark.parametrize("embedded", [False, True])
@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("phased", [False, True])
def test_conversion(tmp_path, sample_file, embedded, zarr_format, workers, phased):
    path = tmp_path / "input.bgen"
    expected = write_bgen(path, embedded=embedded, phased=phased)
    root = bgen.convert(
        path,
        sample_file=sample_file,
        zarr_format=zarr_format,
        variants_chunk_size=3,
        samples_chunk_size=2,
        worker_processes=workers,
    )
    baseline = bgen.convert(
        path, zarr_format=zarr_format, variants_chunk_size=3, samples_chunk_size=2
    )
    nt.assert_array_equal(root["sample_id"][:], ["z", "a", "b", "c"])
    if not phased:
        nt.assert_allclose(root["call_GP"][0], expected)
    assert root.attrs == baseline.attrs
    for name in baseline.array_keys():
        if name != "sample_id":
            nt.assert_array_equal(root[name][:], baseline[name][:])


def test_no_discovery(tmp_path, sample_file):
    path = tmp_path / "example.bgen"
    write_bgen(path)
    sample_file.write_text("malformed\n")
    root = bgen.convert(path)
    nt.assert_array_equal(root["sample_id"][:], [f"person_{j}" for j in range(4)])


@pytest.mark.parametrize("rows", [1, 3, 5])
def test_count_mismatch(tmp_path, sample_file, rows):
    path = tmp_path / "input.bgen"
    write_bgen(path)
    sample_file.write_text("ID_1\n0\n" + "".join(f"s{j}\n" for j in range(rows)))
    out = tmp_path / "output.vcz"
    with pytest.raises(ValueError, match=f"found {rows} sample rows, expected 4"):
        bgen.convert(path, out, sample_file=sample_file)
    assert not out.exists()


@pytest.mark.parametrize("source", ["convert", "BgenFormat"])
def test_option_conflicts_before_reads(tmp_path, source):
    function = getattr(bgen, source)
    args = [tmp_path / "absent.bgen"]
    if source == "BgenFormat":
        args.append(tmp_path / "absent.metafile")
    with mock.patch.object(cbgen, "BgenFile", side_effect=AssertionError("read")):
        with pytest.raises(ValueError, match="mutually exclusive"):
            function(*args, sample_ids=[], sample_file=tmp_path / "absent.sample")
        with pytest.raises(ValueError, match="requires sample_file"):
            function(*args, sample_id_column="ID_2")


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_cli(tmp_path, sample_file, zarr_format):
    path = tmp_path / "input.bgen"
    write_bgen(path, embedded=False)
    out = tmp_path / "output.vcz"
    result = click.testing.CliRunner().invoke(
        cli.bgen2zarr_main,
        [
            "convert",
            str(path),
            str(out),
            "--sample-file",
            str(sample_file),
            "--sample-id-column",
            "ID_2",
            "--zarr-format",
            str(zarr_format),
            "-l",
            "3",
            "-w",
            "2",
            "-Q",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    nt.assert_array_equal(zarr.open_group(out)["sample_id"][:], ["z", "a", "b", "c"])


@pytest.mark.parametrize("failure", ["conflict", "selector", "malformed", "count"])
def test_cli_preserves_force_output(tmp_path, sample_file, failure):
    path = tmp_path / "input.bgen"
    write_bgen(path)
    out = tmp_path / "output.vcz"
    out.mkdir()
    marker = out / "keep"
    marker.write_text("existing data")
    options = ["--sample-file", str(sample_file)]
    if failure == "conflict":
        options += ["--sample-ids", str(sample_file)]
    elif failure == "selector":
        options = ["--sample-id-column", "ID_2"]
    elif failure == "malformed":
        sample_file.write_text("ID_1\n0\nNA\n")
    else:
        sample_file.write_text("ID_1\n0\none\n")
    result = click.testing.CliRunner().invoke(
        cli.bgen2zarr_main,
        ["convert", str(path), str(out), "--force", "-Q", *options],
    )
    assert result.exit_code != 0
    assert marker.read_text() == "existing data"
    assert list(tmp_path.glob(".bio2zarr-bgen*")) == []
    if failure in ("conflict", "selector"):
        assert result.exit_code == 2
    else:
        assert isinstance(result.exception, ValueError)


def test_cli_conflict_before_read(tmp_path, sample_file):
    path = tmp_path / "input.bgen"
    path.touch()
    with mock.patch("builtins.open", side_effect=AssertionError("read")):
        result = click.testing.CliRunner().invoke(
            cli.bgen2zarr_main,
            [
                "convert",
                str(path),
                str(tmp_path / "out"),
                "--sample-file",
                str(sample_file),
                "--sample-ids",
                str(sample_file),
                "--force",
            ],
        )
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output


def test_cli_valid_force_overwrite(tmp_path, sample_file):
    path = tmp_path / "input.bgen"
    write_bgen(path)
    out = tmp_path / "output.vcz"
    out.mkdir()
    (out / "old").write_text("old")
    result = click.testing.CliRunner().invoke(
        cli.bgen2zarr_main,
        [
            "convert",
            str(path),
            str(out),
            "--sample-file",
            str(sample_file),
            "--force",
            "-Q",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert not (out / "old").exists()
    nt.assert_array_equal(zarr.open_group(out)["sample_id"][:], ["z", "a", "b", "c"])
