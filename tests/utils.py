from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import numpy.testing as nt
import xarray as xr
import zarr


def assert_zarr_equal(left, right):
    """Compare schemas, attributes, codecs and values, including NaN payload bits."""
    assert left.attrs.asdict() == right.attrs.asdict()
    assert set(left.array_keys()) == set(right.array_keys())
    for name in left.array_keys():
        a, b = left[name], right[name]
        assert a.attrs.asdict() == b.attrs.asdict(), name
        assert a.dtype == b.dtype, name
        assert a.shape == b.shape, name
        assert a.chunks == b.chunks, name
        assert a.metadata.to_dict() == b.metadata.to_dict(), name
        x, y = a[:], b[:]
        if a.dtype.kind == "f":
            x = x.view(f"u{a.dtype.itemsize}")
            y = y.view(f"u{b.dtype.itemsize}")
        nt.assert_array_equal(x, y, err_msg=name)


def load_dataset(
    store: str | Path | MutableMapping[str, bytes] | zarr.Group,
    storage_options: dict[str, str] | None = None,
    **kwargs: Any,
) -> xr.Dataset:
    """Load an Xarray dataset from Zarr storage."""
    if isinstance(store, zarr.Group):
        store = store.store

    ds: xr.Dataset = xr.open_zarr(
        store,
        storage_options=storage_options,
        concat_characters=False,
        consolidated=False,
        **kwargs,
    )  # type: ignore[no-untyped-call]
    for v in ds:
        # Workaround for https://github.com/pydata/xarray/issues/4386
        if v.endswith("_mask"):  # type: ignore
            ds[v] = ds[v].astype(bool)
    return ds
