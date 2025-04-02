import zarr


def zarr_v3() -> bool:
    return zarr.__version__ >= "3"


if zarr_v3():
    # Use zarr format v2 even when running with zarr-python v3
    ZARR_FORMAT_KWARGS = dict(zarr_format=2)
else:
    ZARR_FORMAT_KWARGS = dict()


# See discussion in https://github.com/zarr-developers/zarr-python/issues/2529
def first_dim_iter(z):
    for chunk in range(z.cdata_shape[0]):
        yield from z.blocks[chunk]
