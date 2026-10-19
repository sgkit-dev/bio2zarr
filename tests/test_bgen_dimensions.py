"""Dimension strategies and checked assumptions for block-first conversion."""

import math
from unittest import mock

import cbgen
import click.testing
import numpy as np
import numpy.testing as nt
import pytest

from bio2zarr import bgen, cli, constants
from bio2zarr.vcz_verification import assert_all_fill_float, assert_all_missing_float
from tests.test_bgen import write_bgen
from tests.test_bgen_general import mixed_records, write_layout1, write_records


@pytest.mark.parametrize(
    "options",
    [
        {"dimension_source": "wrong"},
        {"dimension_source": True},
        {"probability_model": "wrong"},
        {"max_ploidy": True},
        {"max_ploidy": 2.0},
        {"max_ploidy": -1},
        {"max_ploidy": 64},
        {"max_ploidy": 2, "dimension_source": "actual"},
        {"max_ploidy": 2, "dimension_source": "declared"},
    ],
)
def test_dimension_options_before_input_or_output(tmp_path, options):
    out = tmp_path / "out.vcz"
    with pytest.raises(
        ValueError, match="dimension_source|probability_model|max_ploidy"
    ):
        bgen.convert(tmp_path / "missing.bgen", out, **options)
    assert not out.exists()


@pytest.mark.parametrize("compression", [0, 1, 2])
@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("source", ["actual", "declared", "provided"])
def test_dimension_modes_preserve_mixed_data(
    tmp_path, compression, zarr_format, source
):
    records = mixed_records.__wrapped__()
    path = tmp_path / "input.bgen"
    write_records(path, records, compression)
    options = (
        {"max_ploidy": 3} if source == "provided" else {"dimension_source": source}
    )
    root = bgen.convert(
        path,
        zarr_format=zarr_format,
        variants_chunk_size=2,
        samples_chunk_size=1,
        **options,
    )
    for j, record in enumerate(records):
        nt.assert_array_equal(root["call_PLOIDY"][j], record["ploidy"])
        assert root["variant_bgen_phased"][j] == record.get("phased", False)
        field = "call_HP" if record.get("phased", False) else "call_GP"
        missing = record.get("missing", [False] * 4)
        for sample, probability in enumerate(record["probability"]):
            value = root[field][j, sample]
            if missing[sample]:
                expected_bits = np.asarray(constants.FLOAT64_MISSING).view("u8")
                nt.assert_array_equal(
                    value[: len(probability)].view("u8"), expected_bits
                )
            else:
                nt.assert_allclose(
                    value[: len(probability)], np.array(probability) / 255
                )
            assert_all_fill_float(value[len(probability) :])
    assert root.attrs["bio2zarr_bgen_dimensions"]["dimension_source"] == source


@pytest.mark.parametrize("model", ["unphased", "phased", "mixed"])
@pytest.mark.parametrize("compression", [0, 1, 2])
@pytest.mark.parametrize("float_dtype", ["float32", "float64"])
def test_provided_bound_skips_discovery(tmp_path, model, compression, float_dtype):
    path = tmp_path / "input.bgen"
    write_bgen(path, compression=compression, phased=model == "phased")
    decode = cbgen.BgenFile.read_genotypes_batch
    seen = []

    def read(source, offsets, **kwargs):
        assert "max_ncomb" not in kwargs
        assert kwargs["max_actual_ncomb"] == (3 if model == "unphased" else 4)
        seen.extend(offsets)
        return decode(source, offsets, **kwargs)

    with (
        mock.patch.object(
            cbgen.BgenFile, "read_dimensions", side_effect=AssertionError
        ),
        mock.patch.object(
            cbgen.BgenFile, "read_genotype_metadata", side_effect=AssertionError
        ),
        mock.patch.object(cbgen.BgenFile, "get_ncombs", side_effect=AssertionError),
        mock.patch.object(cbgen.BgenFile, "read_genotypes_batch", read),
    ):
        root = bgen.convert(
            path,
            max_ploidy=2,
            probability_model=model,
            variants_chunk_size=2,
            float_dtype=float_dtype,
        )
    assert len(seen) == len(set(seen)) == 7
    assert ("call_GP" in root) == (model != "phased")
    assert ("call_HP" in root) == (model != "unphased")
    if model == "mixed":
        assert_all_missing_float(root["call_HP"][:, :, :1])
        assert_all_fill_float(root["call_HP"][:, :, 1:])


@pytest.mark.parametrize("compression", [0, 1, 2])
@pytest.mark.parametrize("phased", [False, True])
@pytest.mark.parametrize("source", ["actual", "declared", "provided"])
def test_loose_declared_bound_padding(tmp_path, compression, phased, source):
    record = {
        "alleles": ("A", "C", "G"),
        "ploidy": (1, 1),
        "max_ploidy": 4,
        "phased": phased,
        "probability": [[51, 102, 102], [255, 0, 0]],
    }
    path = tmp_path / "input.bgen"
    write_records(path, [record], compression)
    options = (
        {"max_ploidy": 2} if source == "provided" else {"dimension_source": source}
    )
    root = bgen.convert(path, **options)
    p = {"actual": 1, "declared": 4, "provided": 2}[source]
    width = p * 3 if phased else math.comb(p + 2, p)
    field = "call_HP" if phased else "call_GP"
    assert root[field].shape == (1, 2, width)
    nt.assert_allclose(root[field][0, :, :3], np.array(record["probability"]) / 255)
    assert_all_fill_float(root[field][0, :, 3:])
    nt.assert_allclose(root["call_DS"][0], [[0.4, 0.4], [0, 0]])


@pytest.mark.parametrize("compression", [0, 1])
def test_layout1_no_dimension_payload_read(tmp_path, compression):
    path = tmp_path / "input.bgen"
    expected = write_layout1(path, compression)
    with (
        mock.patch.object(
            cbgen.BgenFile, "read_dimensions", side_effect=AssertionError
        ),
        mock.patch.object(
            cbgen.BgenFile, "read_genotype_metadata", side_effect=AssertionError
        ),
    ):
        root = bgen.convert(path)
    nt.assert_array_equal(root["call_GP"][0], expected)


@pytest.mark.parametrize(
    "options",
    [
        {"max_ploidy": 1, "probability_model": "mixed"},
        {"probability_model": "phased"},
    ],
)
def test_layout1_incompatible_assumption(tmp_path, options):
    path = tmp_path / "input.bgen"
    write_layout1(path, 1)
    out = tmp_path / "out.vcz"
    with pytest.raises(ValueError, match="Layout 1"):
        bgen.convert(path, out, **options)
    assert not out.exists()


@pytest.mark.parametrize("assumption", ["ploidy", "phase"])
def test_assumption_violation_does_not_publish(tmp_path, assumption):
    path = tmp_path / "input.bgen"
    records = mixed_records.__wrapped__()
    records.insert(
        0,
        {
            "alleles": ("A", "C", "G", "T"),
            "ploidy": (0, 0, 0, 0),
            "probability": [[255]] * 4,
        },
    )
    write_records(path, records, 1)
    out = tmp_path / "out.vcz"
    # Generous global capacity means the converter, not just native capacity
    # enforcement, must catch a per-sample ploidy/model violation.
    options = (
        {"max_ploidy": 2, "probability_model": "mixed"}
        if assumption == "ploidy"
        else {"max_ploidy": 3, "probability_model": "unphased"}
    )
    with pytest.raises(ValueError, match="sample ploidy exceeds|phase model violates"):
        bgen.convert(path, out, variants_chunk_size=1, **options)
    assert not out.exists()


def test_explicit_bound_overflow_before_output(tmp_path):
    path = tmp_path / "input.bgen"
    out = tmp_path / "out.vcz"
    # Choose sufficiently many alleles to exceed the backend's uint32 width.
    record = {
        "alleles": tuple(f"A{i}" for i in range(16)),
        "ploidy": (0, 0),
        "probability": [[255], [255]],
    }
    write_records(path, [record], 0)
    with pytest.raises(ValueError, match="uint32"):
        bgen.convert(path, out, max_ploidy=63, probability_model="mixed")
    assert not out.exists()


def test_provided_bound_auto_ignores_huge_declared_width(tmp_path):
    path = tmp_path / "input.bgen"
    records = [
        {
            "alleles": tuple(f"A{i}" for i in range(16)),
            "ploidy": (0, 0),
            "max_ploidy": 63,
            "probability": [[255], [255]],
        },
        {
            "alleles": tuple(f"A{i}" for i in range(16)),
            "ploidy": (0, 0),
            "max_ploidy": 63,
            "phased": True,
            "probability": [[], []],
        },
    ]
    write_records(path, records, 1)
    root = bgen.convert(path, max_ploidy=0)
    assert root["call_GP"].shape == (2, 2, 1)
    assert root["call_HP"].shape == (2, 2, 1)
    nt.assert_array_equal(root["call_GP"][0], [[1], [1]])
    assert_all_fill_float(root["call_HP"][1])


def test_old_backend_dimension_interface_error(tmp_path, monkeypatch):
    monkeypatch.delattr(cbgen.BgenFile, "read_dimensions")
    with pytest.raises(ImportError, match="published 4.13.0"):
        bgen.convert(tmp_path / "input.bgen")


@pytest.mark.parametrize("source", ["actual", "declared", "provided"])
@pytest.mark.parametrize("compression", [0, 1, 2])
def test_zero_ploidy_phased_retains_model(tmp_path, source, compression):
    path = tmp_path / "input.bgen"
    record = {
        "alleles": ("A", "C"),
        "ploidy": (0, 0),
        "phased": True,
        "probability": [[], []],
    }
    write_records(path, [record], compression)
    options = (
        {"max_ploidy": 0, "probability_model": "phased"}
        if source == "provided"
        else {"dimension_source": source}
    )
    root = bgen.convert(path, **options)
    assert "call_GP" not in root
    assert root["call_HP"].shape == (1, 2, 1)
    assert_all_fill_float(root["call_HP"][:])
    nt.assert_array_equal(root["call_DS"][:], [[0, 0]])


def test_cli_dimension_options_parity(tmp_path):
    path = tmp_path / "input.bgen"
    write_bgen(path)
    out = tmp_path / "out.vcz"
    with mock.patch.object(bgen, "convert", wraps=bgen.convert) as convert:
        result = click.testing.CliRunner().invoke(
            cli.bgen2zarr_main,
            [
                "convert",
                str(path),
                str(out),
                "--max-ploidy",
                "2",
                "--probability-model",
                "mixed",
                "-Q",
            ],
        )
    assert result.exit_code == 0, result.output
    assert convert.call_args.kwargs["max_ploidy"] == 2
    assert convert.call_args.kwargs["probability_model"] == "mixed"
    assert convert.call_args.kwargs["dimension_source"] is None


def test_cli_conflicting_dimension_options_no_overwrite(tmp_path):
    path = tmp_path / "input.bgen"
    write_bgen(path)
    out = tmp_path / "out.vcz"
    out.mkdir()
    sentinel = out / "user-data"
    sentinel.write_text("keep")
    result = click.testing.CliRunner().invoke(
        cli.bgen2zarr_main,
        [
            "convert",
            str(path),
            str(out),
            "--max-ploidy",
            "2",
            "--dimension-source",
            "actual",
            "--force",
        ],
    )
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output
    assert sentinel.read_text() == "keep"
