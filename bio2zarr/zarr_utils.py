import logging
import os

import zarr

logger = logging.getLogger(__name__)

# Use zarr format v2 by default even when running with zarr-python v3
# NOTE: this interface was introduced for experimentation with zarr
# format 3 and is not envisaged as a long-term interface.
try:
    ZARR_FORMAT = int(os.environ.get("BIO2ZARR_ZARR_FORMAT", "2"))
except Exception:
    ZARR_FORMAT = 2


def zarr_v3() -> bool:
    return zarr.__version__ >= "3"


if zarr_v3():
    ZARR_FORMAT_KWARGS = dict(zarr_format=ZARR_FORMAT)
    # In zarr-python v3 strings are stored as string arrays (T) with itemsize 16
    STRING_DTYPE_NAME = "T"
    STRING_ITEMSIZE = 16
else:
    ZARR_FORMAT_KWARGS = dict()
    # In zarr-python v2 strings are stored as object arrays (O) with itemsize 8
    STRING_DTYPE_NAME = "O"
    STRING_ITEMSIZE = 8


# See discussion in https://github.com/zarr-developers/zarr-python/issues/2529
def first_dim_iter(z):
    for chunk in range(z.cdata_shape[0]):
        yield from z.blocks[chunk]


def zarr_exists(path):
    # NOTE: this is too strict, we should support more general Zarrs, see #276
    return (path / ".zmetadata").exists() or (path / "zarr.json").exists()


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
        array = group.array(
            name,
            data=data,
            shape=shape,
            dtype=dtype,
            compressor=compressor,
            **kwargs,
        )
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
        array = group.empty(
            name=name,
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            compressor=compressor,
            filters=filters,
            **kwargs,
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
        from zarr.codecs.blosc import BloscCodec

        if isinstance(compressor, BloscCodec):
            return compressor._blosc_codec.get_config()
        else:
            return compressor.as_dict()["configuration"]


def _convert_v2_compressor_to_v3_codec(compressor, dtype):
    # import here since this is zarr-python v3 only
    from zarr.core.dtype import parse_dtype
    from zarr.metadata.migrate_v3 import _convert_compressor

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
