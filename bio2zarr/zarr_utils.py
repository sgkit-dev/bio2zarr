import zarr
from packaging.version import Version


def zarr_v3() -> bool:
    return Version(zarr.__version__).major >= 3


if zarr_v3():
    # Use zarr format v2 even when running with zarr-python v3
    ZARR_FORMAT_KWARGS = dict(zarr_format=2)
else:
    ZARR_FORMAT_KWARGS = dict()
