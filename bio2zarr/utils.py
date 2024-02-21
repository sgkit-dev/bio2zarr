import struct
from typing import IO, Any, Dict, Optional, Sequence

import fsspec

from bio2zarr.typing import PathType


def ceildiv(a: int, b: int) -> int:
    """Safe integer ceil function"""
    return -(-a // b)


def get_file_length(
    path: PathType, storage_options: Optional[Dict[str, str]] = None
) -> int:
    """Get the length of a file in bytes."""
    url = str(path)
    storage_options = storage_options or {}
    with fsspec.open(url, **storage_options) as openfile:
        fs = openfile.fs
        size = fs.size(url)
        if size is None:
            raise IOError(f"Cannot determine size of file {url}")  # pragma: no cover
        return int(size)


def get_file_offset(vfp: int) -> int:
    """Convert a block compressed virtual file pointer to a file offset."""
    address_mask = 0xFFFFFFFFFFFF
    return vfp >> 16 & address_mask


def read_bytes_as_value(f: IO[Any], fmt: str, nodata: Optional[Any] = None) -> Any:
    """Read bytes using a `struct` format string and return the unpacked data value.

    Parameters
    ----------
    f : IO[Any]
        The IO stream to read bytes from.
    fmt : str
        A Python `struct` format string.
    nodata : Optional[Any], optional
        The value to return in case there is no further data in the stream, by default None

    Returns
    -------
    Any
        The unpacked data value read from the stream.
    """
    data = f.read(struct.calcsize(fmt))
    if not data:
        return nodata
    values = struct.Struct(fmt).unpack(data)
    assert len(values) == 1
    return values[0]


def read_bytes_as_tuple(f: IO[Any], fmt: str) -> Sequence[Any]:
    """Read bytes using a `struct` format string and return the unpacked data values.

    Parameters
    ----------
    f : IO[Any]
        The IO stream to read bytes from.
    fmt : str
        A Python `struct` format string.

    Returns
    -------
    Sequence[Any]
        The unpacked data values read from the stream.
    """
    data = f.read(struct.calcsize(fmt))
    return struct.Struct(fmt).unpack(data)


def open_gzip(path: PathType, storage_options: Optional[Dict[str, str]]) -> IO[Any]:
    url = str(path)
    storage_options = storage_options or {}
    openfile: IO[Any] = fsspec.open(url, compression="gzip", **storage_options)
    return openfile
