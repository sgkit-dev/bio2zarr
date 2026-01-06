import zarr


def zarr_v3() -> bool:
    return zarr.__version__ >= "3"


if zarr_v3():
    # Use zarr format v2 even when running with zarr-python v3
    ZARR_FORMAT_KWARGS = dict(zarr_format=2)
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
