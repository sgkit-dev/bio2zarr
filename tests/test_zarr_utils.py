import zipfile

import numpy as np
import numpy.testing as nt
import zarr

from bio2zarr import zarr_utils


def _create_test_zarr(path):
    """Create a minimal zarr store at path for testing."""
    root = zarr.open(store=path, mode="w", zarr_format=2)
    root.attrs["test_attr"] = "hello"
    root.create_array("data", data=np.array([1, 2, 3]))
    return root


class TestZipZarr:
    def test_creates_valid_zip(self, tmp_path):
        dir_path = tmp_path / "store"
        zip_path = tmp_path / "store.zip"
        _create_test_zarr(dir_path)
        zarr_utils.zip_zarr(dir_path, zip_path)
        assert zip_path.exists()
        assert dir_path.exists()
        root = zarr.open(zarr.storage.ZipStore(zip_path, mode="r"), mode="r")
        assert root.attrs["test_attr"] == "hello"
        nt.assert_array_equal(root["data"][:], [1, 2, 3])

    def test_uses_zip_stored(self, tmp_path):
        dir_path = tmp_path / "store"
        zip_path = tmp_path / "store.zip"
        _create_test_zarr(dir_path)
        zarr_utils.zip_zarr(dir_path, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                assert info.compress_type == zipfile.ZIP_STORED


class TestDirToMemoryStore:
    def test_copies_data(self, tmp_path):
        dir_path = tmp_path / "store"
        _create_test_zarr(dir_path)
        root = zarr_utils.dir_to_memory_store(dir_path)
        assert isinstance(root.store, zarr.storage.MemoryStore)
        assert root.attrs["test_attr"] == "hello"
        nt.assert_array_equal(root["data"][:], [1, 2, 3])

    def test_mode_parameter(self, tmp_path):
        dir_path = tmp_path / "store"
        _create_test_zarr(dir_path)
        root = zarr_utils.dir_to_memory_store(dir_path, mode="r+")
        assert not root.store.read_only

    def test_default_mode_read_only(self, tmp_path):
        dir_path = tmp_path / "store"
        _create_test_zarr(dir_path)
        root = zarr_utils.dir_to_memory_store(dir_path)
        assert root.read_only
