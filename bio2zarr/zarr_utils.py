import logging
import os
import pathlib
import zipfile

import numpy as np
import zarr

logger = logging.getLogger(__name__)

# Use zarr format v2 by default even when running with zarr-python v3
# NOTE: this interface was introduced for experimentation with zarr
# format 3 and is not envisaged as a long-term interface.
try:
    ZARR_FORMAT = int(os.environ.get("BIO2ZARR_ZARR_FORMAT", "2"))
except Exception:
    ZARR_FORMAT = 2


ZARR_FORMAT_KWARGS = dict(zarr_format=ZARR_FORMAT)
# In zarr-python v3 strings are stored as string arrays (T) with itemsize 16
STRING_DTYPE_NAME = "T"
STRING_ITEMSIZE = 16


# See discussion in https://github.com/zarr-developers/zarr-python/issues/2529
def first_dim_iter(z):
    for chunk in range(z.cdata_shape[0]):
        yield from z.blocks[chunk]


def vcf_zarr_exists(path):
    """Tests if a VCF Zarr store exists at the given path."""
    if (path / ".zgroup").exists() or (path / "zarr.json").exists():
        root = zarr.open(path, mode="r")
        return "vcf_zarr_version" in root.attrs
    else:
        return False


def create_group_array(
    group,
    name,
    *,
    data,
    shape,
    dtype,
    compressor=None,
    dimension_names=None,
    **kwargs,
):
    """Create an array within a group."""
    if ZARR_FORMAT == 2:
        # create_array rejects data together with shape/dtype; when data is
        # provided, let it infer shape/dtype from the array.
        v2_kwargs = {**kwargs}
        if compressor is not None:
            v2_kwargs["compressors"] = [compressor]
        if data is not None:
            np_dtype = np.dtypes.StringDType() if dtype == "str" else dtype
            array = group.create_array(
                name, data=np.asarray(data, dtype=np_dtype), **v2_kwargs
            )
        else:
            array = group.create_array(name, shape=shape, dtype=dtype, **v2_kwargs)
        if dimension_names is not None:
            array.attrs["_ARRAY_DIMENSIONS"] = dimension_names
        return array
    else:
        new_kwargs = {**kwargs}
        if compressor is not None:
            compressors = [_convert_v2_compressor_to_v3_codec(compressor, dtype)]
            # TODO: seems odd that we need to set this
            new_kwargs["compressor"] = "auto"
            new_kwargs["compressors"] = compressors
        return group.array(
            name,
            data=data,
            shape=shape,
            dtype=dtype,
            dimension_names=dimension_names,
            **new_kwargs,
        )


def create_empty_group_array(
    group,
    name,
    *,
    shape,
    dtype,
    chunks,
    compressor=None,
    filters=None,
    dimension_names=None,
    **kwargs,
):
    """Create an empty array within a group."""
    if ZARR_FORMAT == 2:
        v2_kwargs = {**kwargs}
        v2_kwargs.pop("zarr_format", None)
        if compressor is not None:
            v2_kwargs["compressors"] = [compressor]
        array = group.create_array(
            name=name,
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            filters=filters,
            **v2_kwargs,
        )
        if dimension_names is not None:
            array.attrs["_ARRAY_DIMENSIONS"] = dimension_names
        return array
    else:
        new_kwargs = {**kwargs}
        new_kwargs.pop("zarr_format")
        if compressor is not None:
            compressors = [_convert_v2_compressor_to_v3_codec(compressor, dtype)]
            # TODO: seems odd that we need to set this
            new_kwargs["compressor"] = "auto"
            new_kwargs["compressors"] = compressors
        return group.array(
            name=name,
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            dimension_names=dimension_names,
            **new_kwargs,
        )


def get_compressor(array):
    try:
        # zarr format v2: compressor (singular)
        return array.compressor
    except TypeError as e:
        # zarr format v3: compressors (plural)
        compressors = array.compressors
        if len(compressors) > 1:
            raise ValueError(
                f"Only one compressor is supported but found {compressors}"
            ) from e
        return compressors[0] if len(compressors) == 1 else None


def get_compressor_config(array):
    compressor = get_compressor(array)
    if hasattr(compressor, "get_config"):
        return compressor.get_config()
    else:
        from zarr.codecs.blosc import BloscCodec  # noqa: PLC0415

        if isinstance(compressor, BloscCodec):
            return compressor._blosc_codec.get_config()
        else:
            return compressor.as_dict()["configuration"]


def _convert_v2_compressor_to_v3_codec(compressor, dtype):
    # import here since this is zarr-python v3 only
    from zarr.core.dtype import parse_dtype  # noqa: PLC0415
    from zarr.metadata.migrate_v3 import _convert_compressor  # noqa: PLC0415

    return _convert_compressor(compressor, parse_dtype(dtype, zarr_format=3))


def move_chunks(src_path, dest_path, partition, name):
    if ZARR_FORMAT == 2:
        dest = dest_path / name
        chunk_files = [
            path for path in src_path.iterdir() if not path.name.startswith(".")
        ]
    else:
        dest = dest_path / name / "c"
        dest.mkdir(exist_ok=True)
        src_chunks = src_path / "c"
        if not src_chunks.exists():
            chunk_files = []
        else:
            chunk_files = [
                path for path in src_chunks.iterdir() if not path.name.startswith(".")
            ]
    # TODO check for a count of then number of files. If we require a
    # dimension_separator of "/" then we could make stronger assertions
    # here, as we'd always have num_variant_chunks
    logger.debug(f"Moving {len(chunk_files)} chunks for {name} partition {partition}")
    for chunk_file in chunk_files:
        os.rename(chunk_file, dest / chunk_file.name)


def zip_zarr(dir_path, zip_path):
    """Create a zip archive of a zarr directory store."""
    dir_path = pathlib.Path(dir_path)
    zip_path = pathlib.Path(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for file in sorted(dir_path.rglob("*")):
            if file.is_file():
                zf.write(file, file.relative_to(dir_path))


def dir_to_memory_store(dir_path, mode="r"):
    """Copy a zarr directory store into a MemoryStore and return the opened group.

    Uses zarr's async store API (LocalStore.get / MemoryStore.set) to copy data
    between stores. This avoids constructing Buffer objects directly, which would
    require reaching into zarr.core.buffer.cpu — a submodule that isn't part of
    zarr's documented public API. The async approach uses only public store
    interfaces (LocalStore, MemoryStore, default_buffer_prototype) and lets zarr
    handle buffer types internally.

    Zarr 3.x does not yet implement copy_store(), so this manual copy is needed.
    If a public store-to-store copy is added in a future zarr release, this
    function should be replaced with it.
    """
    src = zarr.storage.LocalStore(dir_path, read_only=True)
    dst = zarr.storage.MemoryStore()

    async def _copy():
        proto = zarr.core.buffer.default_buffer_prototype()
        async for key in src.list_prefix(""):
            val = await src.get(key, proto)
            if val is not None:
                await dst.set(key, val)

    zarr.core.sync.sync(_copy())
    return zarr.open(store=dst, mode=mode)
