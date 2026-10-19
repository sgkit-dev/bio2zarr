"""Reuse native sidecars without modifying them or confusing them with BGI."""

import os
import pathlib
from contextlib import contextmanager
from unittest import mock

import cbgen
import numpy as np
import numpy.testing as nt
import pytest

from bio2zarr import bgen
from tests.test_bgen import write_bgen
from tests.test_bgen_general import write_layout1


@pytest.mark.parametrize("layout", [1, 2])
@pytest.mark.parametrize("failure", [None, "metadata", "decode", "write"])
def test_native_handles_closed_before_temporary_cleanup(
    tmp_path, monkeypatch, layout, failure
):
    path = tmp_path / "input.bgen"
    if layout == 1:
        write_layout1(path, 1)
    else:
        write_bgen(path)
    active = set()
    iterators = []
    native_metafile = bgen.native_metafile

    def retain(method):
        def wrapped(*args, **kwargs):
            iterator = method(*args, **kwargs)
            iterators.append(iterator)
            return iterator

        return wrapped

    for name in ("iter_variants", "iter_metadata", "iter_decoded_blocks"):
        monkeypatch.setattr(
            bgen.BgenFormat, name, retain(getattr(bgen.BgenFormat, name))
        )

    @contextmanager
    def checked_metafile(path):
        with native_metafile(path) as index:
            try:
                yield index
            finally:
                assert not active, "Native handles still open at temporary cleanup"

    def track(cls):
        enter, exit = cls.__enter__, cls.__exit__

        def tracked_enter(handle):
            result = enter(handle)
            active.add(id(handle))
            return result

        def tracked_exit(handle, *args):
            try:
                return exit(handle, *args)
            finally:
                active.remove(id(handle))

        return (
            mock.patch.object(cls, "__enter__", tracked_enter),
            mock.patch.object(cls, "__exit__", tracked_exit),
        )

    file_enter, file_exit = track(cbgen.BgenFile)
    index_enter, index_exit = track(cbgen.BgenMetafile)
    target = {
        "metadata": (bgen, "probability_lengths"),
        "decode": (bgen.BgenFormat, "validate_genotype"),
        "write": (bgen.BgenZarrWriter, "call_field"),
    }
    with (
        file_enter,
        file_exit,
        index_enter,
        index_exit,
        mock.patch.object(bgen, "native_metafile", checked_metafile),
    ):
        if failure is None:
            bgen.convert(path, variants_chunk_size=1)
        else:
            obj, method = target[failure]
            with (
                mock.patch.object(obj, method, side_effect=ValueError("injected")),
                pytest.raises(ValueError, match="injected"),
            ):
                bgen.convert(path, variants_chunk_size=1)
    assert not active


@pytest.mark.parametrize("layout", [1, 2])
@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("invalid", [False, True])
def test_temporary_index_cleanup_with_workers(tmp_path, layout, workers, invalid):
    path = tmp_path / "input.bgen"
    if layout == 1:
        probabilities = np.array([[32768, 32768, 0]], dtype="<u2") if invalid else None
        write_layout1(path, 1, probabilities=probabilities)
    else:
        write_bgen(path, invalid=invalid)
    created = []
    create = cbgen.BgenFile.create_metafile

    def record(source, index_path):
        created.append(pathlib.Path(index_path))
        return create(source, index_path)

    out = tmp_path / "out.vcz"
    with mock.patch.object(cbgen.BgenFile, "create_metafile", record):
        if invalid:
            with pytest.raises(ValueError, match="invalid genotype probabilities"):
                bgen.convert(path, out, worker_processes=workers, variants_chunk_size=1)
            assert not out.exists()
        else:
            bgen.convert(path, out, worker_processes=workers, variants_chunk_size=1)
            assert out.exists()
    assert len(created) == 1
    assert not created[0].parent.exists()
    assert not list(tmp_path.glob(".bio2zarr-bgen-*"))


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
