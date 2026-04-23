import logging
import os
import pathlib
import zipfile

import numcodecs
import numpy as np
import tqdm
import zarr
from numpy.dtypes import StringDType
from zarr.codecs.blosc import BloscCodec, BloscShuffle

logger = logging.getLogger(__name__)

numcodecs.blosc.use_threads = False

# Storage format (zarr v2 vs v3) is chosen at runtime in the CLI via
# --zarr-format, in the API by zarr_format, or by the BIO2ZARR_DEFAULT_ZARR_FORMAT
# env var. If specified in the CLI or API these override any env var setting.
# The default is Zarr format version 2. The underlying zarr-python library is
# always v3 (>=3.1); this flag only controls which on-disk format we write.
try:
    DEFAULT_ZARR_FORMAT = int(os.environ.get("BIO2ZARR_DEFAULT_ZARR_FORMAT", "2"))
except ValueError:
    DEFAULT_ZARR_FORMAT = 2


# In zarr-python v3 strings are stored as string arrays (T) with itemsize 16
STRING_DTYPE_NAME = "T"
STRING_ITEMSIZE = 16


# Canonical, format-independent compressor configs (numcodecs-style dicts).
# Stored verbatim in schema JSON so a schema written under one Zarr format
# stays readable under the other.
DEFAULT_COMPRESSOR_CONFIG = {
    "id": "blosc",
    "cname": "zstd",
    "clevel": 7,
    "shuffle": numcodecs.Blosc.SHUFFLE,
    "blocksize": 0,
}
DEFAULT_COMPRESSOR_BOOL_CONFIG = {
    "id": "blosc",
    "cname": "zstd",
    "clevel": 7,
    "shuffle": numcodecs.Blosc.BITSHUFFLE,
    "blocksize": 0,
}
DEFAULT_COMPRESSOR_GENOTYPES_CONFIG = {
    "id": "blosc",
    "cname": "zstd",
    "clevel": 7,
    "shuffle": numcodecs.Blosc.BITSHUFFLE,
    "blocksize": 0,
}

_BLOSC_SHUFFLE_V3 = {
    numcodecs.Blosc.NOSHUFFLE: BloscShuffle.noshuffle,
    numcodecs.Blosc.SHUFFLE: BloscShuffle.shuffle,
    numcodecs.Blosc.BITSHUFFLE: BloscShuffle.bitshuffle,
}


def make_compressor(config, zarr_format):
    """Build a format-correct compressor from a numcodecs-style config dict.

    Returns a numcodecs codec under zarr_format==2 and a zarr v3 codec under
    zarr_format==3. Only Blosc is supported, since that is the only
    compressor bio2zarr produces.
    """
    if config.get("id") != "blosc":
        raise NotImplementedError(
            f"Only blosc compressors are supported, got {config!r}"
        )
    if zarr_format == 2:
        return numcodecs.get_codec(config)
    return BloscCodec(
        cname=config["cname"],
        clevel=config["clevel"],
        shuffle=_BLOSC_SHUFFLE_V3[config.get("shuffle", numcodecs.Blosc.SHUFFLE)],
        blocksize=config.get("blocksize", 0),
    )


# See discussion in https://github.com/zarr-developers/zarr-python/issues/2529
def first_dim_iter(z):
    for chunk in range(z.cdata_shape[0]):
        yield from z.blocks[chunk]


def open_zarr_append(store, zarr_format):
    """Open a Zarr store for append using the given Zarr format"""
    zarr_format_kwargs = dict(zarr_format=zarr_format or DEFAULT_ZARR_FORMAT)
    return zarr.open(store=store, mode="a", **zarr_format_kwargs)


def open_vcf_zarr(path):
    """Open a VCF Zarr store at ``path``, returning the root group.

    ``path`` may be a directory store or a ``.zip`` archive. Raises
    ``ValueError`` if the path does not hold a VCF Zarr store.
    """
    path = pathlib.Path(path)
    store = path
    if path.suffix == ".zip":
        store = zarr.storage.ZipStore(path, mode="r")
    try:
        root = zarr.open(store, mode="r")
    except (OSError, zipfile.BadZipFile) as e:
        raise ValueError(f"Not in VcfZarr format: {path}") from e
    if "vcf_zarr_version" not in root.attrs:
        raise ValueError(f"Not in VcfZarr format: {path}")
    return root


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
    new_kwargs = {**kwargs}
    if compressor is not None:
        new_kwargs["compressors"] = [
            make_compressor(compressor, group.metadata.zarr_format)
        ]

    # Zarr format v2 rejects dimension_names on create_array; we instead
    # write the xarray _ARRAY_DIMENSIONS attribute after the fact.
    if group.metadata.zarr_format == 3:
        new_kwargs.pop("zarr_format", None)
        new_kwargs["dimension_names"] = dimension_names

    # create_array rejects data together with shape/dtype; when data is
    # provided, let it infer shape/dtype from the array.
    if data is not None:
        array = group.create_array(
            name, data=np.asarray(data, dtype=dtype), **new_kwargs
        )
    else:
        array = group.create_array(name, shape=shape, dtype=dtype, **new_kwargs)
    if group.metadata.zarr_format == 2 and dimension_names is not None:
        array.attrs["_ARRAY_DIMENSIONS"] = dimension_names
    return array


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
    new_kwargs = {**kwargs}
    new_kwargs.pop("zarr_format", None)
    if compressor is not None:
        new_kwargs["compressors"] = [
            make_compressor(compressor, group.metadata.zarr_format)
        ]
    if group.metadata.zarr_format == 2:
        # Filter list is interpreted as numcodecs-style config dicts and
        # converted here so callers don't need to import numcodecs.
        # Zarr v2 also requires a VLenUTF8 filter to accompany "T"-dtype
        # arrays whenever an explicit compressor is supplied; inject it
        # here so string-array creation is transparent to callers.
        codecs = []
        if filters is not None:
            codecs = [numcodecs.get_codec(f) for f in filters]
        if dtype == STRING_DTYPE_NAME or dtype == StringDType():
            codecs.append(numcodecs.VLenUTF8())
        if len(codecs) == 0:
            codecs = None
        new_kwargs["filters"] = codecs
    else:
        # Zarr v3 uses its own ArrayArrayCodec objects for filters; the
        # numcodecs filters (e.g. VLenUTF8) are not applicable because v3
        # has native handling for variable-length strings.
        new_kwargs["dimension_names"] = dimension_names

    array = group.create_array(
        name=name, shape=shape, dtype=dtype, chunks=chunks, **new_kwargs
    )
    if group.metadata.zarr_format == 2 and dimension_names is not None:
        array.attrs["_ARRAY_DIMENSIONS"] = dimension_names
    return array


def get_compressor(array):
    compressors = array.compressors
    if len(compressors) > 1:
        raise ValueError(f"Only one compressor is supported but found {compressors}")
    return compressors[0] if len(compressors) == 1 else None


def get_compressor_config(array):
    compressor = get_compressor(array)
    if compressor is None:
        return None
    # numcodecs codecs (zarr format v2 path) expose get_config directly.
    if hasattr(compressor, "get_config"):
        return compressor.get_config()
    # Zarr v3's BloscCodec wraps a numcodecs.Blosc instance at _blosc_codec;
    # reach through to get the same dict shape as the v2 path.
    if isinstance(compressor, BloscCodec):
        return compressor._blosc_codec.get_config()
    raise TypeError(f"Unsupported compressor type: {type(compressor).__name__}")


def move_chunks(src_path, dest_path, partition, name, zarr_format):
    # Zarr v2 stores chunk files directly in the array directory; v3 places
    # them under a c/ subdirectory.
    if zarr_format == 2:
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
    logger.debug(f"Moving {len(chunk_files)} chunks for {name} partition {partition}")
    for chunk_file in chunk_files:
        os.rename(chunk_file, dest / chunk_file.name)


def zip_zarr(dir_path, zip_path, *, show_progress=False):
    """Create a zip archive of a zarr directory store."""
    dir_path = pathlib.Path(dir_path)
    zip_path = pathlib.Path(zip_path)
    files = sorted(p for p in dir_path.rglob("*") if p.is_file())
    logger.info(f"Zipping {len(files)} files from {dir_path} to {zip_path}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for file in tqdm.tqdm(
            files, desc="     Zip", unit=" files", disable=not show_progress
        ):
            zf.write(file, file.relative_to(dir_path))


def unzip_zarr(zip_path, dir_path, *, show_progress=False):
    """Extract a zip archive of a zarr directory store."""
    zip_path = pathlib.Path(zip_path)
    dir_path = pathlib.Path(dir_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        logger.info(f"Unzipping {len(members)} files from {zip_path} to {dir_path}")
        for member in tqdm.tqdm(
            members, desc="   Unzip", unit=" files", disable=not show_progress
        ):
            zf.extract(member, dir_path)


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
