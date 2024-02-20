import dataclasses
import contextlib
import concurrent.futures as cf
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

    @property
    def chunk_length(self):
        return self.buff.shape[0]

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


class ThreadedZarrEncoder(contextlib.AbstractContextManager):
    def __init__(self, buffered_arrays, encoder_threads):
        self.buffered_arrays = buffered_arrays
        self.executor = cf.ThreadPoolExecutor(max_workers=encoder_threads)
        self.chunk_length = buffered_arrays[0].chunk_length
        assert all(ba.chunk_length == self.chunk_length for ba in self.buffered_arrays)
        self.futures = []
        self.array_offset = 0
        self.next_row = -1

    def next_buffer_row(self):
        self.next_row += 1
        if self.next_row == self.chunk_length:
            self.swap_buffers()
            self.array_offset += self.chunk_length
            self.next_row = 0
        return self.next_row

    def wait_on_futures(self):
        for future in cf.as_completed(self.futures):
            exception = future.exception()
            if exception is not None:
                raise exception

    def swap_buffers(self):
        self.wait_on_futures()
        self.futures = []
        for ba in self.buffered_arrays:
            # TODO add debug log
            # print("Scheduling", ba.array, offset, buff_stop)
            self.futures.extend(
                ba.async_flush(self.executor, self.array_offset, self.next_row)
            )
            ba.swap_buffers()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Normal exit condition
            self.next_row += 1
            self.swap_buffers()
            self.wait_on_futures()
        # TODO add arguments to wait and cancel_futures appropriate
        # for the an error condition occuring here. Generally need
        # to think about the error exit condition here (like running
        # out of disk space) to see what the right behaviour is.
        self.executor.shutdown()
        return False
