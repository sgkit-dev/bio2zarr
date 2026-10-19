"""Block-first decoding invariants, independent of numerical fixture coverage."""

import weakref
from unittest import mock

import cbgen
import numpy as np
import numpy.testing as nt
import pytest

from bio2zarr import bgen
from tests.test_bgen import write_bgen
from tests.test_bgen_general import mixed_records, write_layout1, write_records


def assert_identical(left, right):
    assert set(left.array_keys()) == set(right.array_keys())
    for name in left.array_keys():
        actual, expected = left[name][:], right[name][:]
        if actual.dtype.kind == "f":
            # Include dtype-specific missing/fill NaN payloads.
            nt.assert_array_equal(
                actual.view(f"u{actual.itemsize}"),
                expected.view(f"u{expected.itemsize}"),
            )
        else:
            nt.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("layout", "compression"), [(1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]
)
@pytest.mark.parametrize("zarr_format", [2, 3])
def test_all_fields_share_one_decode(tmp_path, layout, compression, zarr_format):
    path = tmp_path / "input.bgen"
    if layout == 1:
        write_layout1(path, compression, nvariants=7)
    else:
        write_records(path, mixed_records.__wrapped__(), compression)
    with cbgen.BgenFile(path) as source:
        count = source.nvariants
    decode = cbgen.BgenFile.read_genotypes_batch
    metadata = cbgen.BgenFile.read_genotype_metadata
    seen = []
    scanned = []
    events = []

    def read(source, offsets, *, max_ncomb, precision):
        assert precision == 64
        assert max_ncomb is not None
        seen.extend(offsets)
        events.append("decode")
        return decode(source, offsets, max_ncomb=max_ncomb, precision=precision)

    def scan(source, offsets):
        scanned.extend(offsets)
        events.append("metadata")
        return metadata(source, offsets)

    with (
        mock.patch.object(cbgen.BgenFile, "read_genotypes_batch", read),
        mock.patch.object(cbgen.BgenFile, "read_genotype_metadata", scan),
        mock.patch.object(cbgen.BgenFile, "read_genotype", side_effect=AssertionError),
        mock.patch.object(bgen.BgenFormat, "iter_field", side_effect=AssertionError),
    ):
        root = bgen.convert(
            path,
            variants_chunk_size=2,
            samples_chunk_size=1,
            zarr_format=zarr_format,
        )
    assert len(seen) == len(set(seen)) == count
    assert scanned == seen
    assert events == sorted(events, key=lambda event: event == "decode")
    assert root["call_DS"].shape[0] == count


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("float_dtype", ["float32", "float64"])
def test_memory_subdivision_preserves_chunks(tmp_path, zarr_format, float_dtype):
    path = tmp_path / "input.bgen"
    write_bgen(path)
    options = dict(
        variants_chunk_size=3,
        samples_chunk_size=1,
        zarr_format=zarr_format,
        float_dtype=float_dtype,
    )
    expected = bgen.convert(path, **options)
    decode = cbgen.BgenFile.read_genotypes_batch
    counts = []

    def read(source, offsets, **kwargs):
        counts.append(len(offsets))
        return decode(source, offsets, **kwargs)

    with (
        mock.patch.object(bgen, "BLOCK_MEMORY", 700),
        mock.patch.object(cbgen.BgenFile, "read_genotypes_batch", read),
    ):
        actual = bgen.convert(path, tmp_path / "out.vcz", **options)
    assert counts == [1] * 7
    assert_identical(actual, expected)
    assert actual["call_GP"].chunks == expected["call_GP"].chunks


def test_single_variant_exceeds_budget(tmp_path):
    path = tmp_path / "input.bgen"
    write_bgen(path)
    out = tmp_path / "out.vcz"
    with (
        mock.patch.object(bgen, "BLOCK_MEMORY", 1),
        mock.patch.object(cbgen.BgenFile, "read_genotypes_batch") as decode,
        pytest.raises(ValueError, match="buffer budget"),
    ):
        bgen.convert(path, out)
    decode.assert_not_called()
    assert not out.exists()
    assert not list(tmp_path.glob(".bio2zarr-bgen-*"))


def test_previous_batch_released_before_next_decode(tmp_path):
    path = tmp_path / "input.bgen"
    write_records(path, mixed_records.__wrapped__(), 1)
    decode = cbgen.BgenFile.read_genotypes_batch
    previous = None

    def read(source, offsets, **kwargs):
        nonlocal previous
        assert previous is None or previous() is None
        result = decode(source, offsets, **kwargs)
        previous = weakref.ref(result.probability)
        return result

    with mock.patch.object(cbgen.BgenFile, "read_genotypes_batch", read):
        bgen.convert(path, variants_chunk_size=1, samples_chunk_size=1)
    assert previous() is None


@pytest.mark.parametrize("suffix", [".vcz", ".vcz.zip"])
def test_invalid_later_block_does_not_publish(tmp_path, suffix):
    path = tmp_path / "input.bgen"
    write_bgen(path)
    out = tmp_path / f"out{suffix}"
    decode = cbgen.BgenFile.read_genotypes_batch
    calls = 0

    def read(source, offsets, **kwargs):
        nonlocal calls
        calls += 1
        result = decode(source, offsets, **kwargs)
        if calls == 2:
            result.probability[0, 0, 0] = -1
        return result

    with (
        mock.patch.object(cbgen.BgenFile, "read_genotypes_batch", read),
        pytest.raises(ValueError, match="invalid genotype probabilities"),
    ):
        bgen.convert(path, out, variants_chunk_size=1)
    assert calls == 2
    assert not out.exists()
    assert not list(tmp_path.glob(".bio2zarr-bgen-*"))


def test_backend_without_batch_fails_before_output(tmp_path):
    path = tmp_path / "input.bgen"
    write_bgen(path)
    out = tmp_path / "out.vcz"
    with (
        mock.patch.object(
            bgen, "hasattr", side_effect=lambda obj, name: name == "layout", create=True
        ),
        pytest.raises(ImportError, match="native.*batch interface"),
    ):
        bgen.convert(path, out)
    assert not out.exists()


@pytest.mark.parametrize("phased", [False, True])
@pytest.mark.parametrize("all_missing", [False, True])
def test_declared_ploidy_does_not_inflate_output(tmp_path, phased, all_missing):
    path = tmp_path / "input.bgen"
    record = {
        "alleles": ("A", "C", "G"),
        "ploidy": (1, 1),
        "max_ploidy": 4,
        "phased": phased,
        "missing": (all_missing, all_missing),
        "probability": [[51, 102, 102], [255, 0, 0]],
    }
    write_records(path, [record], 2)
    root = bgen.convert(path, samples_chunk_size=1)
    field = "call_HP" if phased else "call_GP"
    assert root[field].shape == (1, 2, 3)
    nt.assert_array_equal(root["call_PLOIDY"][:], [[1, 1]])
    if all_missing:
        assert np.isnan(root[field][:]).all()
        assert np.isnan(root["call_DS"][:]).all()
    else:
        nt.assert_allclose(root[field][0], np.array(record["probability"]) / 255)
        nt.assert_allclose(root["call_DS"][0], [[0.4, 0.4], [0, 0]])
