"""Required exporter-to-importer interoperability tests, in a dedicated environment."""

import hashlib
import itertools
import math
import pathlib
import sqlite3
import subprocess
import sys
from unittest import mock

import cbgen
import numpy as np
import numpy.testing as nt
import pytest
import zarr
from vcztools import VczReader
from vcztools import bgen as exporter

from bio2zarr import bgen, constants
from tests.test_bgen_general import mixed_records, write_records

DATA = pathlib.Path(__file__).resolve().parents[1] / "tests" / "data" / "bgen"
SAMPLE_IDS = ["sample_three", "sample_one", "sample_\u03b2", "sample_two"]
METADATA = (
    "sample_id",
    "contig_id",
    "variant_contig",
    "variant_position",
    "variant_allele",
    "variant_id",
    "variant_id_mask",
    "variant_bgen_id",
    "variant_bgen_phased",
    "call_PLOIDY",
    "call_BGEN_MISSING",
)


def run_cli(module, *args, success=True):
    result = subprocess.run(
        [sys.executable, "-m", module, *map(str, args)],
        capture_output=True,
        text=True,
        check=False,
    )
    if success:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0, result.stdout + result.stderr
    return result


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns


def independent_dosage(root):
    """Derive dosage without either package's dosage implementation."""
    ploidies = root["call_PLOIDY"][:]
    result = np.full(root["call_DS"].shape, constants.FLOAT64_FILL)
    if result.ndim == 2:
        result = result[:, :, None]
    for row, phased in enumerate(root["variant_bgen_phased"][:]):
        alleles = sum(allele != "" for allele in root["variant_allele"][row])
        for sample, ploidy in enumerate(ploidies[row]):
            p = int(ploidy)
            if root["call_BGEN_MISSING"][row, sample]:
                result[row, sample] = constants.FLOAT64_MISSING
            elif phased:
                values = root["call_HP"][row, sample, : p * alleles]
                result[row, sample, : alleles - 1] = values.reshape(p, alleles).sum(
                    axis=0
                )[1:]
            else:
                genotypes = sorted(
                    itertools.combinations_with_replacement(range(alleles), p),
                    key=lambda genotype: genotype[::-1],
                )
                probabilities = root["call_GP"][row, sample, : len(genotypes)]
                counts = np.array(
                    [
                        [genotype.count(a) for a in range(1, alleles)]
                        for genotype in genotypes
                    ]
                )
                result[row, sample, : alleles - 1] = probabilities @ counts
    return result.reshape(root["call_DS"].shape)


def assert_roundtrip(expected, actual, bits, selected=None):
    assert "call_genotype" not in expected
    assert "call_genotype" not in actual
    if selected is None:
        selected = list(range(expected["sample_id"].shape[0]))
    for name in METADATA:
        value = expected[name][:]
        if name == "sample_id":
            value = value[selected]
        elif name.startswith("call_"):
            value = value[:, selected]
        nt.assert_array_equal(actual[name][:], value, err_msg=name)
    assert actual.attrs == expected.attrs
    scale = (1 << bits) - 1
    for field in ("call_GP", "call_HP", "call_DS"):
        if field not in expected:
            assert field not in actual
            continue
        before = expected[field][:][:, selected]
        after = actual[field][:]
        if before.ndim == 3:
            before = before[:, :, : after.shape[-1]]
        nt.assert_array_equal(np.isnan(after), np.isnan(before), err_msg=field)
        nan = np.isnan(before)
        nt.assert_array_equal(
            after.view("u8")[nan], before.view("u8")[nan], err_msg=field
        )
        for row, phased in enumerate(expected["variant_bgen_phased"][:]):
            if field == "call_DS":
                p = int(expected["call_PLOIDY"][row, selected].max())
                width = expected["call_HP" if phased else "call_GP"].shape[-1]
                bound = p / scale if phased else p * width / scale
            else:
                bound = 1 / scale
            finite = np.isfinite(before[row])
            nt.assert_allclose(
                after[row][finite],
                before[row][finite],
                rtol=0,
                atol=bound + 1e-12,
                err_msg=f"{field}, variant {row}, {bits} bits",
            )
    oracle = independent_dosage(actual)
    nt.assert_allclose(actual["call_DS"][:], oracle, rtol=0, atol=1e-12)
    nan = np.isnan(oracle)
    nt.assert_array_equal(actual["call_DS"][:].view("u8")[nan], oracle.view("u8")[nan])


def verify_indexes_and_import(stem, target, zarr_format, *, sample_file=True):
    payload = stem.with_suffix(".bgen")
    sample = stem.with_suffix(".sample")
    native = pathlib.Path(str(payload) + ".metafile")
    bgi = pathlib.Path(str(payload) + ".bgi")
    assert payload.is_file()
    assert native.is_file()
    assert bgi.is_file()
    before = {path: fingerprint(path) for path in (native, bgi)}
    with sqlite3.connect(bgi) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    options = ["--sample-file", sample] if sample_file else []
    run_cli(
        "bio2zarr",
        "bgen2zarr",
        "convert",
        payload,
        target,
        "--zarr-format",
        zarr_format,
        *options,
    )
    with mock.patch.object(
        cbgen.BgenFile,
        "create_metafile",
        side_effect=AssertionError("existing native index was regenerated"),
    ):
        bgen.convert(payload, sample_file=sample if sample_file else None)
    assert {path: fingerprint(path) for path in before} == before
    if sample_file:
        assert sample.is_file()
    return zarr.open_group(target, mode="r")


@pytest.fixture(params=["complex", "haplotypes", "mixed8", "mixed23"])
def source_bgen(request, tmp_path):
    name = request.param
    if name == "complex":
        return DATA / "complex.23bits.bgen", {}
    if name == "haplotypes":
        return DATA / "haplotypes.bgen", {}
    path = tmp_path / f"{name}.bgen"
    write_records(
        path,
        mixed_records.__wrapped__(),
        compression=2,
        bits=23 if name == "mixed23" else 8,
    )
    return path, {"sample_ids": SAMPLE_IDS}


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("bits", [1, 16, 32])
def test_cli_probability_roundtrip(tmp_path, source_bgen, zarr_format, bits):
    path, options = source_bgen
    source = bgen.convert(
        path,
        tmp_path / "source.vcz",
        zarr_format=zarr_format,
        variants_chunk_size=3,
        samples_chunk_size=2,
        **options,
    )
    stem = tmp_path / "exported"
    run_cli(
        "vcztools",
        "view-bgen",
        tmp_path / "source.vcz",
        "-o",
        stem,
        "--metafile",
        "--no-header-samples",
        "--bits-per-probability",
        bits,
    )
    payload = stem.with_suffix(".bgen")
    native = pathlib.Path(str(payload) + ".metafile")
    with cbgen.BgenFile(payload) as reader, cbgen.BgenMetafile(str(native)) as index:
        offsets = [
            variant.offset
            for partition in range(index.npartitions)
            for variant in index.read_partition(partition).variants
        ]
        nt.assert_array_equal(reader.read_genotype_metadata(offsets).nbits, bits)
        assert not reader.contain_samples
    assert bgen.read_sample_file(stem.with_suffix(".sample")) == list(
        source["sample_id"][:]
    )
    actual = verify_indexes_and_import(stem, tmp_path / "actual.vcz", zarr_format)
    assert_roundtrip(source, actual, bits)
    if bits == 16 and options:
        assert any(
            np.any(
                np.abs(
                    source[field][:][np.isfinite(source[field][:])] * 65535
                    - np.round(source[field][:][np.isfinite(source[field][:])] * 65535)
                )
                > 1e-5
            )
            for field in ("call_GP", "call_HP")
        ) == (path.stem == "mixed23")


@pytest.fixture
def probability_source(tmp_path):
    path = tmp_path / "independent.bgen"
    records = [
        {
            "alleles": ("A", "C", "G"),
            "ploidy": (3, 1, 0, 2),
            "missing": (False, False, False, True),
            "probability": [
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 210],
                [38, 64, 153],
                [255],
                [0] * 6,
            ],
        },
        {
            "alleles": ("A", "C", "G"),
            "ploidy": (3, 1, 0, 2),
            "phased": True,
            "missing": (False, False, True, True),
            "probability": [
                [26, 51, 178, 77, 51, 127, 26, 102, 127],
                [51, 77, 127],
                [],
                [0] * 6,
            ],
        },
    ]
    for record in records:
        for p, values in zip(record["ploidy"], record["probability"]):
            assert len(values) == (
                p * 3 if record.get("phased") else math.comb(p + 2, 2)
            )
    write_records(path, records, compression=1)
    return path


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("subset", [False, True])
def test_probability_subset_roundtrip(
    tmp_path, probability_source, zarr_format, subset
):
    source = bgen.convert(
        probability_source,
        zarr_format=zarr_format,
        sample_ids=["0", "sample_one", "sample_\u03b2", "sample_two"],
    )
    selected = [1, 3] if subset else list(range(4))
    stem = tmp_path / "subset"
    with VczReader(source) as reader:
        reader.set_sample_indexes(selected)
        exporter.write_bgen(
            reader,
            stem.with_suffix(".bgen"),
            sample_path=stem.with_suffix(".sample"),
            bgi_path=pathlib.Path(str(stem) + ".bgen.bgi"),
            metafile_path=str(stem) + ".bgen.metafile",
            embed_header_samples=False,
            bits_per_probability=16,
        )
    actual = verify_indexes_and_import(stem, tmp_path / "actual.vcz", zarr_format)
    assert_roundtrip(source, actual, 16, selected)


def test_embedded_na_roundtrip(tmp_path, probability_source):
    source = bgen.convert(
        probability_source, sample_ids=["NA", "0", "sample_\u03b2", "sample_two"]
    )
    stem = tmp_path / "embedded"
    with VczReader(source) as reader:
        exporter.write_bgen(
            reader,
            stem.with_suffix(".bgen"),
            bgi_path=pathlib.Path(str(stem) + ".bgen.bgi"),
            metafile_path=str(stem) + ".bgen.metafile",
            bits_per_probability=16,
        )
    actual = verify_indexes_and_import(
        stem, tmp_path / "actual.vcz", 2, sample_file=False
    )
    assert_roundtrip(source, actual, 16)


def test_na_sample_preflight_preserves_existing_output(tmp_path, probability_source):
    source = tmp_path / "source.vcz"
    bgen.convert(probability_source, source, sample_ids=["NA", "b", "c", "d"])
    stem = tmp_path / "protected"
    paths = [
        stem.with_suffix(".bgen"),
        stem.with_suffix(".sample"),
        pathlib.Path(str(stem) + ".bgen.bgi"),
        pathlib.Path(str(stem) + ".bgen.metafile"),
    ]
    result = run_cli(
        "vcztools", "view-bgen", source, "-o", stem, "--metafile", success=False
    )
    assert "NA" in result.stderr
    assert not any(path.exists() for path in paths)
    for path in paths:
        path.write_bytes(b"existing-output")
    before = {path: fingerprint(path) for path in paths}
    result = run_cli(
        "vcztools", "view-bgen", source, "-o", stem, "--metafile", success=False
    )
    assert {path: fingerprint(path) for path in paths} == before
