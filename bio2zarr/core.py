import dataclasses
import logging

import zarr
import numpy as np


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class BufferedArray:
    array: zarr.Array
    buff: np.ndarray

    def __init__(self, array):
        self.array = array
        dims = list(array.shape)
        dims[0] = min(array.chunks[0], array.shape[0])
        self.buff = np.zeros(dims, dtype=array.dtype)

    def swap_buffers(self):
        self.buff = np.zeros_like(self.buff)

    def async_flush(self, executor, offset, buff_stop=None):
        return async_flush_array(executor, self.buff[:buff_stop], self.array, offset)


def sync_flush_array(np_buffer, zarr_array, offset):
    zarr_array[offset : offset + np_buffer.shape[0]] = np_buffer


def async_flush_array(executor, np_buffer, zarr_array, offset):
    """
    Flush the specified chunk aligned buffer to the specified zarr array.
    """
    logger.debug(f"Schedule flush {zarr_array} @ {offset}")
    assert zarr_array.shape[1:] == np_buffer.shape[1:]
    # print("sync", zarr_array, np_buffer)

    if len(np_buffer.shape) == 1:
        futures = [executor.submit(sync_flush_array, np_buffer, zarr_array, offset)]
    else:
        futures = async_flush_2d_array(executor, np_buffer, zarr_array, offset)
    return futures


def async_flush_2d_array(executor, np_buffer, zarr_array, offset):
    # Flush each of the chunks in the second dimension separately
    s = slice(offset, offset + np_buffer.shape[0])

    def flush_chunk(start, stop):
        zarr_array[s, start:stop] = np_buffer[:, start:stop]

    chunk_width = zarr_array.chunks[1]
    zarr_array_width = zarr_array.shape[1]
    start = 0
    futures = []
    while start < zarr_array_width:
        stop = min(start + chunk_width, zarr_array_width)
        future = executor.submit(flush_chunk, start, stop)
        futures.append(future)
        start = stop

    return futures
