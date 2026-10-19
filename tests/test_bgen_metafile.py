"""Reuse native sidecars without modifying them or confusing them with BGI."""

import os
import pathlib
from unittest import mock

import cbgen
import numpy.testing as nt
import pytest

from bio2zarr import bgen
from tests.test_bgen import write_bgen
from tests.test_bgen_general import write_layout1


@pytest.mark.parametrize("layout", [1, 2])
@pytest.mark.parametrize("workers", [0, 2])
def test_reuse_existing_native_metafile(tmp_path, layout, workers):
    path = tmp_path / "input.bgen"
    if layout == 1:
        write_layout1(path, 1)
    else:
        write_bgen(path)
    expected = bgen.convert(path)
    index_path = path.with_name(path.name + ".metafile")
    with cbgen.BgenFile(path) as source:
        source.create_metafile(str(index_path))
    original = index_path.read_bytes()
    index_path.chmod(0o444)
    timestamp = index_path.stat().st_mtime_ns
    with (
        mock.patch.object(
            cbgen.BgenFile, "create_metafile", side_effect=AssertionError
        ),
        mock.patch.object(
            bgen.tempfile, "TemporaryDirectory", wraps=bgen.tempfile.TemporaryDirectory
        ) as temporary,
    ):
        actual = bgen.convert(path, worker_processes=workers)
    assert all(
        call.kwargs.get("prefix") != "bio2zarr-bgen-"
        for call in temporary.call_args_list
    )
    for name in actual.array_keys():
        nt.assert_array_equal(actual[name][:], expected[name][:])
    assert index_path.read_bytes() == original
    assert index_path.stat().st_mtime_ns == timestamp


def test_missing_native_metafile_is_temporary(tmp_path):
    path = tmp_path / "input.bgen"
    write_bgen(path)
    create = cbgen.BgenFile.create_metafile
    created = []

    def record(source, index_path):
        created.append(index_path)
        return create(source, index_path)

    with mock.patch.object(cbgen.BgenFile, "create_metafile", record):
        bgen.convert(path)
    assert len(created) == 1
    index = pathlib.Path(created[0])
    assert index.parent != path.parent
    assert not index.exists()
    assert not path.with_name(path.name + ".metafile").exists()


@pytest.mark.parametrize("failure", ["stale", "wrong_count", "corrupt"])
def test_bad_existing_index_is_not_replaced(tmp_path, failure):
    path = tmp_path / "input.bgen"
    write_bgen(path)
    index_path = path.with_name(path.name + ".metafile")
    if failure == "wrong_count":
        other = tmp_path / "other.bgen"
        write_bgen(other, nvariants=3)
        source_path = other
    else:
        source_path = path
    with cbgen.BgenFile(source_path) as source:
        source.create_metafile(str(index_path))
    if failure == "stale":
        timestamp = path.stat().st_mtime_ns - 1_000_000_000
        os.utime(index_path, ns=(timestamp, timestamp))
    elif failure == "corrupt":
        index_path.write_bytes(b"truncated")
    original = index_path.read_bytes()
    out = tmp_path / "out.vcz"
    error, message = {
        "stale": (ValueError, "older than the input"),
        "wrong_count": (ValueError, "different variant count"),
        "corrupt": (RuntimeError, "Failed to open metafile"),
    }[failure]
    with (
        mock.patch.object(
            cbgen.BgenFile, "create_metafile", side_effect=AssertionError
        ),
        pytest.raises(error, match=message),
    ):
        bgen.convert(path, out)
    assert not out.exists()
    assert index_path.read_bytes() == original


def test_bgi_is_not_mistaken_for_native_metafile(tmp_path):
    path = tmp_path / "input.bgen"
    write_bgen(path)
    bgi = path.with_name(path.name + ".bgi")
    bgi.write_bytes(b"SQLite format 3\x00")
    original = bgi.read_bytes()
    create = cbgen.BgenFile.create_metafile
    with mock.patch.object(
        cbgen.BgenFile, "create_metafile", autospec=True, side_effect=create
    ) as generated:
        bgen.convert(path)
    assert generated.call_count == 1
    assert bgi.read_bytes() == original


@pytest.mark.parametrize("npartitions", [0, 8])
def test_invalid_partition_count_is_rejected(tmp_path, npartitions):
    path = tmp_path / "input.bgen"
    write_bgen(path)
    index_path = path.with_name(path.name + ".metafile")
    with cbgen.BgenFile(path) as source:
        source.create_metafile(str(index_path))
    with mock.patch.object(cbgen, "BgenMetafile") as open_index:
        index = open_index.return_value.__enter__.return_value
        index.nvariants = 7
        index.npartitions = npartitions
        with pytest.raises(ValueError, match="invalid partition count"):
            bgen.convert(path)
        index.read_partition.assert_not_called()
