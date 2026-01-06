from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import xarray as xr


def load_dataset(
    store: str | Path | MutableMapping[str, bytes],
    storage_options: dict[str, str] | None = None,
    **kwargs: Any,
) -> xr.Dataset:
    """Load an Xarray dataset from Zarr storage."""

    ds: xr.Dataset = xr.open_zarr(
        store, storage_options=storage_options, concat_characters=False, **kwargs
    )  # type: ignore[no-untyped-call]
    for v in ds:
        # Workaround for https://github.com/pydata/xarray/issues/4386
        if v.endswith("_mask"):  # type: ignore
            ds[v] = ds[v].astype(bool)
    return ds
