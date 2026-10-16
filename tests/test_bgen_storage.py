import logging
from unittest import mock

import click.testing
import numpy as np
import numpy.testing as nt
import pytest
import zarr

from bio2zarr import bgen, cli, constants, zarr_utils
from tests.test_bgen import write_bgen
from tests.test_bgen_general import mixed_records, write_layout1, write_records


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("float_dtype", ["float32", "float64"])
@pytest.mark.parametrize("compressor", ["zstd", "lz4"])
def test_storage_precision_and_codec(tmp_path, zarr_format, float_dtype, compressor):
    path = tmp_path / "mixed.bgen"
    write_records(path, mixed_records.__wrapped__(), 2, bits=32)
    baseline = bgen.convert(path)
    root = bgen.convert(
        path,
        tmp_path / "out.vcz",
        float_dtype=float_dtype,
        compressor=compressor,
        compression_level=3,
        zarr_format=zarr_format,
        variants_chunk_size=3,
        samples_chunk_size=3,
        worker_processes=2,
    )
    for name in ("call_GP", "call_HP", "call_DS"):
        actual, expected = root[name][:], baseline[name][:]
        assert actual.dtype == np.dtype(float_dtype)
        nt.assert_allclose(actual, expected, rtol=1e-6)
        for old, new in zip(
            constants.FLOAT_MISSING_FILL[expected.dtype],
            constants.FLOAT_MISSING_FILL[actual.dtype],
        ):
            mask = expected.view("u8") == np.asarray(old).view("u8")
            assert mask.any()
            unsigned = f"u{actual.dtype.itemsize}"
            nt.assert_array_equal(
                actual[mask].view(unsigned), np.asarray(new).view(unsigned)
            )
        codec = zarr_utils.get_compressor(root[name])
        config = (
            codec.get_config() if zarr_format == 2 else codec.to_dict()["configuration"]
        )
        assert config["cname"] == compressor
        assert config["clevel"] == 3
        assert config["shuffle"] == (1 if zarr_format == 2 else "shuffle")
        assert root[name].chunks[-1] == root[name].shape[-1]
    assert root.attrs["bio2zarr_bgen"] == {
        "probabilities": True,
        "dosage": True,
        "float_dtype": float_dtype,
        "compressor": compressor,
        "compression_level": 3,
    }


@pytest.mark.parametrize(("probabilities", "dosage"), [(True, False), (False, True)])
def test_field_selection_and_passes(tmp_path, probabilities, dosage, caplog):
    path = tmp_path / "mixed.bgen"
    write_records(path, mixed_records.__wrapped__(), 1)
    seen = []
    original = bgen.BgenFormat.iter_field

    def read(source, field_name, *args):
        seen.append(field_name)
        yield from original(source, field_name, *args)

    with (
        caplog.at_level(logging.INFO, logger="bio2zarr.bgen"),
        mock.patch.object(bgen.BgenFormat, "iter_field", read),
        mock.patch.object(bgen, "compute_dosage", wraps=bgen.compute_dosage) as compute,
    ):
        root = bgen.convert(path, probabilities=probabilities, dosage=dosage)
    for name in ("GP", "HP"):
        assert (f"call_{name}" in root) == probabilities
        assert (name in seen) == probabilities
    assert ("call_DS" in root) == dosage
    assert ("DS" in seen) == dosage
    assert bool(compute.call_count) == dosage
    for name in (
        "call_PLOIDY",
        "call_BGEN_MISSING",
        "variant_bgen_phased",
        "variant_position",
        "variant_allele",
        "variant_bgen_id",
        "sample_id",
    ):
        assert name in root
    assert "call_genotype" not in root
    assert "BGEN storage:" in caplog.text
    assert root.attrs["bio2zarr_bgen"]["probabilities"] == probabilities
    assert root.attrs["bio2zarr_bgen"]["dosage"] == dosage


def test_float64_layout1_defaults(tmp_path):
    path = tmp_path / "layout1.bgen"
    expected = write_layout1(path, 1)
    root = bgen.convert(path)
    nt.assert_array_equal(root["call_GP"][0], expected)
    assert root["call_GP"].dtype == np.dtype("float64")
    assert root["call_GP"][0, 2].sum() == 32767 / 32768
    codec = zarr_utils.get_compressor(root["call_GP"])
    config = (
        codec.get_config()
        if root.metadata.zarr_format == 2
        else codec.to_dict()["configuration"]
    )
    assert config["cname"] == "zstd"
    assert config["clevel"] == 7
    assert config["shuffle"] == (1 if root.metadata.zarr_format == 2 else "shuffle")


def test_dosage_before_downcast(tmp_path):
    path = tmp_path / "precision.bgen"
    write_records(
        path,
        [
            {
                "alleles": ("A", "T"),
                "ploidy": (2,),
                "probability": [[0, 1, 254]],
            }
        ],
        2,
        bits=32,
    )
    original = bgen.convert(path)["call_DS"][:]
    expected = original.copy()
    expected[np.isnan(expected)] = 0
    expected = expected.astype("float32")
    expected[np.isnan(original)] = constants.FLOAT32_MISSING
    with mock.patch.object(
        bgen, "compute_dosage", wraps=bgen.compute_dosage
    ) as compute:
        root = bgen.convert(path, probabilities=False, float_dtype="float32")
    assert compute.call_count
    assert all(
        call.args[0].probability.dtype == np.dtype("float64")
        for call in compute.call_args_list
    )
    nt.assert_array_equal(root["call_DS"][:], expected)
    rounded_gp = np.array([0, 1, 254], dtype="float64").astype("float32") / np.float32(
        255
    )
    premature = rounded_gp[1] + np.float32(2) * rounded_gp[2]
    assert root["call_DS"][0, 0] != premature


@pytest.mark.parametrize(
    "kwargs",
    [
        {"probabilities": False, "dosage": False},
        {"probabilities": 1},
        {"dosage": None},
        {"float_dtype": "float16"},
        {"float_dtype": "f4"},
        {"compressor": "gzip"},
        {"compression_level": -1},
        {"compression_level": 10},
        {"compression_level": 1.5},
        {"compression_level": True},
    ],
)
def test_invalid_options_before_input_or_output(tmp_path, kwargs):
    out = tmp_path / "out.vcz"
    with mock.patch.object(
        bgen, "BgenFormat", side_effect=AssertionError("opened input")
    ):
        with pytest.raises(ValueError, match="must"):
            bgen.convert(tmp_path / "input.bgen", out, **kwargs)
    assert not out.exists()


@pytest.mark.parametrize(("probabilities", "dosage"), [(True, False), (False, True)])
def test_cli_api_parity(tmp_path, probabilities, dosage):
    path = tmp_path / "mixed.bgen"
    write_records(path, mixed_records.__wrapped__(), 1)
    options = dict(
        probabilities=probabilities,
        dosage=dosage,
        float_dtype="float32",
        compressor="lz4",
        compression_level=0,
        zarr_format=3,
    )
    expected = bgen.convert(path, **options)
    out = tmp_path / "cli.vcz"
    result = click.testing.CliRunner().invoke(
        cli.bgen2zarr_main,
        [
            "convert",
            str(path),
            str(out),
            "--probabilities" if probabilities else "--no-probabilities",
            "--dosage" if dosage else "--no-dosage",
            "--float-dtype",
            "float32",
            "--compressor",
            "lz4",
            "--compression-level",
            "0",
            "--zarr-format",
            "3",
            "-Q",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    root = zarr.open_group(out)
    assert set(root.array_keys()) == set(expected.array_keys())
    assert root.attrs["bio2zarr_bgen"] == expected.attrs["bio2zarr_bgen"]
    for name in root.array_keys():
        a, e = root[name][:], expected[name][:]
        if a.dtype.kind == "f":
            nt.assert_array_equal(a.view("u4"), e.view("u4"))
        else:
            nt.assert_array_equal(a, e)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--no-probabilities", "--no-dosage"],
        ["--float-dtype", "float16"],
        ["--compressor", "gzip"],
        ["--compression-level", "10"],
    ],
)
def test_cli_invalid_options(tmp_path, arguments):
    path = tmp_path / "input.bgen"
    write_bgen(path)
    out = tmp_path / "out.vcz"
    result = click.testing.CliRunner().invoke(
        cli.bgen2zarr_main,
        ["convert", str(path), str(out), *arguments],
    )
    assert result.exit_code == 2
    assert not out.exists()
