import re
import zipfile

import numcodecs
import numpy as np
import numpy.testing as nt
import pytest
import zarr
from zarr.codecs.blosc import BloscCodec, BloscShuffle

from bio2zarr import zarr_utils


@pytest.fixture(params=[2, 3])
def zarr_format(request, monkeypatch):
    monkeypatch.setattr(zarr_utils, "ZARR_FORMAT", request.param)
    return request.param


@pytest.fixture
def group(tmp_path, zarr_format):
    return zarr.open_group(tmp_path / "store", mode="w", zarr_format=zarr_format)


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

    @pytest.mark.parametrize("show_progress", [True, False])
    def test_show_progress(self, tmp_path, capsys, show_progress):
        dir_path = tmp_path / "store"
        zip_path = tmp_path / "store.zip"
        _create_test_zarr(dir_path)
        zarr_utils.zip_zarr(dir_path, zip_path, show_progress=show_progress)
        assert zip_path.exists()
        captured = capsys.readouterr()
        if show_progress:
            assert captured.err != ""
        else:
            assert captured.err == ""


class TestUnzipZarr:
    def test_roundtrip(self, tmp_path):
        dir_path = tmp_path / "store"
        zip_path = tmp_path / "store.zip"
        out_path = tmp_path / "store-out"
        _create_test_zarr(dir_path)
        zarr_utils.zip_zarr(dir_path, zip_path)
        zarr_utils.unzip_zarr(zip_path, out_path)
        root = zarr.open(store=out_path, mode="r")
        assert root.attrs["test_attr"] == "hello"
        nt.assert_array_equal(root["data"][:], [1, 2, 3])

    def test_matches_source_tree(self, tmp_path):
        dir_path = tmp_path / "store"
        zip_path = tmp_path / "store.zip"
        out_path = tmp_path / "store-out"
        _create_test_zarr(dir_path)
        zarr_utils.zip_zarr(dir_path, zip_path)
        zarr_utils.unzip_zarr(zip_path, out_path)
        src_files = sorted(
            p.relative_to(dir_path) for p in dir_path.rglob("*") if p.is_file()
        )
        out_files = sorted(
            p.relative_to(out_path) for p in out_path.rglob("*") if p.is_file()
        )
        assert src_files == out_files

    @pytest.mark.parametrize("show_progress", [True, False])
    def test_show_progress(self, tmp_path, capsys, show_progress):
        dir_path = tmp_path / "store"
        zip_path = tmp_path / "store.zip"
        out_path = tmp_path / "store-out"
        _create_test_zarr(dir_path)
        zarr_utils.zip_zarr(dir_path, zip_path)
        capsys.readouterr()
        zarr_utils.unzip_zarr(zip_path, out_path, show_progress=show_progress)
        assert out_path.exists()
        captured = capsys.readouterr()
        if show_progress:
            assert captured.err != ""
        else:
            assert captured.err == ""


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


class TestMakeCompressor:
    def test_v2_returns_numcodecs(self, monkeypatch):
        monkeypatch.setattr(zarr_utils, "ZARR_FORMAT", 2)
        c = zarr_utils.make_compressor(zarr_utils.DEFAULT_COMPRESSOR_CONFIG)
        assert isinstance(c, numcodecs.Blosc)
        assert c.get_config() == zarr_utils.DEFAULT_COMPRESSOR_CONFIG

    def test_v3_returns_blosc_codec(self, monkeypatch):
        monkeypatch.setattr(zarr_utils, "ZARR_FORMAT", 3)
        c = zarr_utils.make_compressor(zarr_utils.DEFAULT_COMPRESSOR_CONFIG)
        assert isinstance(c, BloscCodec)
        assert c.cname.value == "zstd"
        assert c.clevel == 7
        assert c.shuffle == BloscShuffle.shuffle
        assert c.blocksize == 0

    @pytest.mark.parametrize(
        ("numcodecs_shuffle", "v3_shuffle"),
        [
            (numcodecs.Blosc.NOSHUFFLE, BloscShuffle.noshuffle),
            (numcodecs.Blosc.SHUFFLE, BloscShuffle.shuffle),
            (numcodecs.Blosc.BITSHUFFLE, BloscShuffle.bitshuffle),
        ],
    )
    def test_v3_shuffle_mapping(self, monkeypatch, numcodecs_shuffle, v3_shuffle):
        monkeypatch.setattr(zarr_utils, "ZARR_FORMAT", 3)
        c = zarr_utils.make_compressor(
            {
                "id": "blosc",
                "cname": "zstd",
                "clevel": 5,
                "shuffle": numcodecs_shuffle,
            }
        )
        assert c.shuffle == v3_shuffle
        assert c.blocksize == 0

    def test_v3_default_shuffle_when_omitted(self, monkeypatch):
        monkeypatch.setattr(zarr_utils, "ZARR_FORMAT", 3)
        c = zarr_utils.make_compressor({"id": "blosc", "cname": "lz4", "clevel": 1})
        assert c.shuffle == BloscShuffle.shuffle

    def test_non_blosc_raises(self, zarr_format):
        with pytest.raises(NotImplementedError, match="Only blosc"):
            zarr_utils.make_compressor({"id": "zlib", "level": 1})

    def test_missing_id_raises(self, zarr_format):
        with pytest.raises(NotImplementedError, match="Only blosc"):
            zarr_utils.make_compressor({"cname": "zstd"})


class TestDefaultCompressorConstants:
    def test_default_config(self):
        assert zarr_utils.DEFAULT_COMPRESSOR_CONFIG == {
            "id": "blosc",
            "cname": "zstd",
            "clevel": 7,
            "shuffle": numcodecs.Blosc.SHUFFLE,
            "blocksize": 0,
        }

    def test_default_bool_config(self):
        assert (
            zarr_utils.DEFAULT_COMPRESSOR_BOOL_CONFIG["shuffle"]
            == numcodecs.Blosc.BITSHUFFLE
        )

    def test_default_genotypes_config(self):
        assert (
            zarr_utils.DEFAULT_COMPRESSOR_GENOTYPES_CONFIG["shuffle"]
            == numcodecs.Blosc.BITSHUFFLE
        )


class TestFirstDimIter:
    def test_iterates_in_order(self, tmp_path):
        root = zarr.open_group(tmp_path / "store", mode="w", zarr_format=2)
        data = np.arange(24).reshape(6, 4)
        a = root.create_array("x", data=data, chunks=(2, 4))
        collected = np.stack(list(zarr_utils.first_dim_iter(a)))
        nt.assert_array_equal(collected, data)


class TestOpenVcfZarr:
    def _make_vcf_zarr_dir(self, path, zarr_format):
        root = zarr.open_group(path, mode="w", zarr_format=zarr_format)
        root.attrs["vcf_zarr_version"] = "0.2"
        root.create_array("data", data=np.array([1, 2, 3]))
        return root

    def test_dir_with_attr_returns_root(self, tmp_path, zarr_format):
        path = tmp_path / "store"
        self._make_vcf_zarr_dir(path, zarr_format)
        opened = zarr_utils.open_vcf_zarr(path)
        assert opened.attrs["vcf_zarr_version"] == "0.2"
        nt.assert_array_equal(opened["data"][:], [1, 2, 3])

    def test_dir_accepts_string_path(self, tmp_path, zarr_format):
        path = tmp_path / "store"
        self._make_vcf_zarr_dir(path, zarr_format)
        opened = zarr_utils.open_vcf_zarr(str(path))
        assert opened.attrs["vcf_zarr_version"] == "0.2"

    def test_dir_without_attr(self, tmp_path, zarr_format):
        path = tmp_path / "store"
        zarr.open_group(path, mode="w", zarr_format=zarr_format)
        with pytest.raises(
            ValueError, match=f"Not in VcfZarr format: {re.escape(str(path))}"
        ):
            zarr_utils.open_vcf_zarr(path)

    def test_dir_wrong_attr_value_is_still_accepted(self, tmp_path, zarr_format):
        # The helper only checks for presence, not value. Document that.
        path = tmp_path / "store"
        root = zarr.open_group(path, mode="w", zarr_format=zarr_format)
        root.attrs["vcf_zarr_version"] = ""
        opened = zarr_utils.open_vcf_zarr(path)
        assert "vcf_zarr_version" in opened.attrs

    def test_missing_dir_path(self, tmp_path):
        missing = tmp_path / "nope"
        with pytest.raises(
            ValueError,
            match=f"Not in VcfZarr format: {re.escape(str(missing))}",
        ) as exc_info:
            zarr_utils.open_vcf_zarr(missing)
        assert exc_info.value.__cause__ is not None

    def test_empty_dir(self, tmp_path):
        path = tmp_path / "empty"
        path.mkdir()
        with pytest.raises(ValueError, match="Not in VcfZarr format"):
            zarr_utils.open_vcf_zarr(path)

    def test_regular_file_non_zip(self, tmp_path):
        path = tmp_path / "not_zarr.txt"
        path.write_text("just a text file")
        with pytest.raises(ValueError, match="Not in VcfZarr format"):
            zarr_utils.open_vcf_zarr(path)

    def test_zip_with_attr_returns_root(self, tmp_path, zarr_format):
        dir_path = tmp_path / "store"
        zip_path = tmp_path / "store.vcz.zip"
        self._make_vcf_zarr_dir(dir_path, zarr_format)
        zarr_utils.zip_zarr(dir_path, zip_path)
        opened = zarr_utils.open_vcf_zarr(zip_path)
        assert opened.attrs["vcf_zarr_version"] == "0.2"
        nt.assert_array_equal(opened["data"][:], [1, 2, 3])
        assert isinstance(opened.store, zarr.storage.ZipStore)

    def test_zip_accepts_string_path(self, tmp_path, zarr_format):
        dir_path = tmp_path / "store"
        zip_path = tmp_path / "store.vcz.zip"
        self._make_vcf_zarr_dir(dir_path, zarr_format)
        zarr_utils.zip_zarr(dir_path, zip_path)
        opened = zarr_utils.open_vcf_zarr(str(zip_path))
        assert opened.attrs["vcf_zarr_version"] == "0.2"

    def test_zip_zarr_without_attr(self, tmp_path, zarr_format):
        dir_path = tmp_path / "store"
        zip_path = tmp_path / "store.zip"
        zarr.open_group(dir_path, mode="w", zarr_format=zarr_format)
        zarr_utils.zip_zarr(dir_path, zip_path)
        with pytest.raises(ValueError, match="Not in VcfZarr format"):
            zarr_utils.open_vcf_zarr(zip_path)

    def test_zip_no_zarr_inside(self, tmp_path):
        # Valid zip file, but contains no zarr metadata at all.
        zip_path = tmp_path / "bogus.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("hello.txt", "not a zarr store")
        with pytest.raises(
            ValueError,
            match=f"Not in VcfZarr format: {re.escape(str(zip_path))}",
        ) as exc_info:
            zarr_utils.open_vcf_zarr(zip_path)
        assert exc_info.value.__cause__ is not None

    def test_missing_zip_path(self, tmp_path):
        missing = tmp_path / "nope.zip"
        with pytest.raises(
            ValueError,
            match=f"Not in VcfZarr format: {re.escape(str(missing))}",
        ) as exc_info:
            zarr_utils.open_vcf_zarr(missing)
        assert isinstance(exc_info.value.__cause__, FileNotFoundError)

    def test_not_a_real_zip_file(self, tmp_path):
        bad = tmp_path / "bad.zip"
        bad.write_text("this is not a zip file at all")
        with pytest.raises(
            ValueError, match=f"Not in VcfZarr format: {re.escape(str(bad))}"
        ) as exc_info:
            zarr_utils.open_vcf_zarr(bad)
        assert isinstance(exc_info.value.__cause__, zipfile.BadZipFile)

    def test_dir_with_zip_suffix(self, tmp_path):
        # Pathological: a directory whose name ends with ``.zip``.
        path = tmp_path / "looks_like.zip"
        path.mkdir()
        with pytest.raises(
            ValueError, match=f"Not in VcfZarr format: {re.escape(str(path))}"
        ):
            zarr_utils.open_vcf_zarr(path)


class TestCreateGroupArray:
    def test_data_path_round_trip(self, group):
        a = zarr_utils.create_group_array(
            group,
            "x",
            data=[1, 2, 3, 4],
            shape=4,
            dtype=np.int32,
            dimension_names=["samples"],
        )
        nt.assert_array_equal(a[:], [1, 2, 3, 4])
        assert a.dtype == np.int32

    def test_shape_path(self, group):
        a = zarr_utils.create_group_array(
            group,
            "x",
            data=None,
            shape=(3, 5),
            dtype=np.float64,
            dimension_names=["variants", "samples"],
        )
        assert a.shape == (3, 5)
        assert a.dtype == np.float64

    def test_dimension_names_stored(self, group, zarr_format):
        a = zarr_utils.create_group_array(
            group,
            "x",
            data=[1, 2, 3],
            shape=3,
            dtype=np.int32,
            dimension_names=["variants"],
        )
        if zarr_format == 2:
            assert a.attrs["_ARRAY_DIMENSIONS"] == ["variants"]
        else:
            assert a.metadata.dimension_names == ("variants",)

    def test_no_dimension_names(self, group, zarr_format):
        a = zarr_utils.create_group_array(
            group, "x", data=[1, 2, 3], shape=3, dtype=np.int32
        )
        if zarr_format == 2:
            assert "_ARRAY_DIMENSIONS" not in a.attrs
        else:
            assert a.metadata.dimension_names is None

    def test_compressor_dict_converted(self, group, zarr_format):
        a = zarr_utils.create_group_array(
            group,
            "x",
            data=[1, 2, 3],
            shape=3,
            dtype=np.int32,
            compressor={"id": "blosc", "cname": "lz4", "clevel": 3, "shuffle": 1},
        )
        assert len(a.compressors) == 1
        expected_type = numcodecs.Blosc if zarr_format == 2 else BloscCodec
        assert isinstance(a.compressors[0], expected_type)


class TestCreateEmptyGroupArray:
    def test_compressor_dict_converted(self, group, zarr_format):
        a = zarr_utils.create_empty_group_array(
            group,
            "x",
            shape=(4,),
            dtype=np.int32,
            chunks=(2,),
            compressor={"id": "blosc", "cname": "lz4", "clevel": 3, "shuffle": 1},
        )
        assert len(a.compressors) == 1
        expected_type = numcodecs.Blosc if zarr_format == 2 else BloscCodec
        assert isinstance(a.compressors[0], expected_type)

    def test_no_filters_non_string(self, group, zarr_format):
        a = zarr_utils.create_empty_group_array(
            group,
            "x",
            shape=(4,),
            dtype=np.int32,
            chunks=(2,),
            compressor=zarr_utils.DEFAULT_COMPRESSOR_CONFIG,
        )
        assert a.shape == (4,)
        assert a.dtype == np.int32
        if zarr_format == 2:
            assert a.metadata.filters is None

    def test_user_filters_non_string_v2(self, tmp_path, monkeypatch):
        monkeypatch.setattr(zarr_utils, "ZARR_FORMAT", 2)
        root = zarr.open_group(tmp_path / "s", mode="w", zarr_format=2)
        a = zarr_utils.create_empty_group_array(
            root,
            "x",
            shape=(8,),
            dtype="<i4",
            chunks=(4,),
            compressor=zarr_utils.DEFAULT_COMPRESSOR_CONFIG,
            filters=[{"id": "delta", "dtype": "<i4"}],
        )
        assert len(a.metadata.filters) == 1
        assert isinstance(a.metadata.filters[0], numcodecs.Delta)

    def test_string_dtype_no_user_filters(self, group, zarr_format):
        a = zarr_utils.create_empty_group_array(
            group,
            "x",
            shape=(4,),
            dtype=zarr_utils.STRING_DTYPE_NAME,
            chunks=(2,),
            compressor=zarr_utils.DEFAULT_COMPRESSOR_CONFIG,
        )
        if zarr_format == 2:
            assert len(a.metadata.filters) == 1
            assert isinstance(a.metadata.filters[0], numcodecs.VLenUTF8)
        a[:] = ["a", "b", "c", "d"]
        nt.assert_array_equal(a[:], ["a", "b", "c", "d"])

    def test_string_dtype_with_user_filters_v2(self, tmp_path, monkeypatch):
        monkeypatch.setattr(zarr_utils, "ZARR_FORMAT", 2)
        root = zarr.open_group(tmp_path / "s", mode="w", zarr_format=2)
        # Place a VLenUTF8 in the user filter list; the helper should
        # still append its own VLenUTF8 (the conversion path does not
        # deduplicate). This exercises the "user filters + string dtype"
        # branch where both conversion and injection happen.
        a = zarr_utils.create_empty_group_array(
            root,
            "x",
            shape=(4,),
            dtype=zarr_utils.STRING_DTYPE_NAME,
            chunks=(2,),
            compressor=zarr_utils.DEFAULT_COMPRESSOR_CONFIG,
            filters=[{"id": "vlen-utf8"}],
        )
        assert len(a.metadata.filters) == 2
        assert all(isinstance(f, numcodecs.VLenUTF8) for f in a.metadata.filters)

    def test_dimension_names(self, group, zarr_format):
        a = zarr_utils.create_empty_group_array(
            group,
            "x",
            shape=(3, 2),
            dtype=np.int32,
            chunks=(3, 2),
            compressor=zarr_utils.DEFAULT_COMPRESSOR_CONFIG,
            dimension_names=["variants", "samples"],
        )
        if zarr_format == 2:
            assert a.attrs["_ARRAY_DIMENSIONS"] == ["variants", "samples"]
        else:
            assert a.metadata.dimension_names == ("variants", "samples")

    def test_chunks(self, group):
        a = zarr_utils.create_empty_group_array(
            group,
            "x",
            shape=(10,),
            dtype=np.int32,
            chunks=(4,),
            compressor=zarr_utils.DEFAULT_COMPRESSOR_CONFIG,
        )
        assert a.chunks == (4,)


class TestGetCompressor:
    def test_none_when_no_compressor(self, group):
        a = group.create_array("x", shape=(3,), dtype=np.int32, compressors=None)
        assert zarr_utils.get_compressor(a) is None

    def test_single_compressor(self, group):
        a = zarr_utils.create_empty_group_array(
            group,
            "x",
            shape=(4,),
            dtype=np.int32,
            chunks=(2,),
            compressor=zarr_utils.DEFAULT_COMPRESSOR_CONFIG,
        )
        c = zarr_utils.get_compressor(a)
        assert c is not None

    def test_multiple_compressors_raises(self):
        class Stub:
            compressors = (object(), object())

        with pytest.raises(ValueError, match="Only one compressor"):
            zarr_utils.get_compressor(Stub())


class TestGetCompressorConfig:
    def test_numcodecs_codec(self):
        class Stub:
            compressors = (numcodecs.Blosc(cname="zstd", clevel=5),)

        config = zarr_utils.get_compressor_config(Stub())
        assert config["id"] == "blosc"
        assert config["cname"] == "zstd"
        assert config["clevel"] == 5

    def test_v3_blosc_codec(self):
        codec = BloscCodec(cname="zstd", clevel=4, shuffle=BloscShuffle.shuffle)

        class Stub:
            compressors = (codec,)

        config = zarr_utils.get_compressor_config(Stub())
        assert config["id"] == "blosc"
        assert config["cname"] == "zstd"
        assert config["clevel"] == 4

    def test_unsupported_type_raises(self):
        class FakeCodec:
            pass

        class Stub:
            compressors = (FakeCodec(),)

        with pytest.raises(TypeError, match="Unsupported compressor type"):
            zarr_utils.get_compressor_config(Stub())


class TestMoveChunks:
    def test_v2_layout(self, tmp_path, monkeypatch):
        monkeypatch.setattr(zarr_utils, "ZARR_FORMAT", 2)
        src = tmp_path / "src"
        src.mkdir()
        (src / ".zarray").write_text("{}")
        (src / "0").write_text("chunk0")
        (src / "1").write_text("chunk1")

        dest_root = tmp_path / "dest"
        (dest_root / "arr").mkdir(parents=True)

        zarr_utils.move_chunks(src, dest_root, partition=0, name="arr")

        assert (dest_root / "arr" / "0").read_text() == "chunk0"
        assert (dest_root / "arr" / "1").read_text() == "chunk1"
        # Hidden file is left behind.
        assert (src / ".zarray").exists()

    def test_v3_layout(self, tmp_path, monkeypatch):
        monkeypatch.setattr(zarr_utils, "ZARR_FORMAT", 3)
        src = tmp_path / "src"
        (src / "c").mkdir(parents=True)
        (src / "c" / "0").write_text("chunk0")
        (src / "c" / "1").write_text("chunk1")
        (src / "c" / ".hidden").write_text("ignore")

        dest_root = tmp_path / "dest"
        (dest_root / "arr").mkdir(parents=True)

        zarr_utils.move_chunks(src, dest_root, partition=0, name="arr")

        assert (dest_root / "arr" / "c" / "0").read_text() == "chunk0"
        assert (dest_root / "arr" / "c" / "1").read_text() == "chunk1"
        assert (src / "c" / ".hidden").exists()

    def test_v3_missing_c_directory(self, tmp_path, monkeypatch):
        monkeypatch.setattr(zarr_utils, "ZARR_FORMAT", 3)
        src = tmp_path / "src"
        src.mkdir()

        dest_root = tmp_path / "dest"
        (dest_root / "arr").mkdir(parents=True)

        # Should not raise even though src/c/ does not exist.
        zarr_utils.move_chunks(src, dest_root, partition=0, name="arr")
        assert list((dest_root / "arr" / "c").iterdir()) == []
