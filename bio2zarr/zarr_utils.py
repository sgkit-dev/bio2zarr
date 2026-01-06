import os

import zarr

# Use zarr format v2 by default even when running with zarr-python v3
try:
    ZARR_FORMAT = int(os.environ.get("BIO2ZARR_ZARR_FORMAT", "2"))
except Exception:
    ZARR_FORMAT = 2


def zarr_v3() -> bool:
    return zarr.__version__ >= "3"


if zarr_v3():
    ZARR_FORMAT_KWARGS = dict(zarr_format=ZARR_FORMAT)
    STRING_DTYPE_NAME = "T"
else:
    ZARR_FORMAT_KWARGS = dict()
    STRING_DTYPE_NAME = "O"


# See discussion in https://github.com/zarr-developers/zarr-python/issues/2529
def first_dim_iter(z):
    for chunk in range(z.cdata_shape[0]):
        yield from z.blocks[chunk]


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


def _convert_v2_compressor_to_v3_codec(compressor, dtype):
    # import here since this is zarr-python v3 only
    from zarr.core.dtype import parse_dtype
    from zarr.metadata.migrate_v3 import _convert_compressor

    return _convert_compressor(compressor, parse_dtype(dtype, zarr_format=3))
