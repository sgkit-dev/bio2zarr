import dataclasses
import contextlib
import concurrent.futures as cf
import multiprocessing
import threading
import logging
import functools
import time

import zarr
import numpy as np
import tqdm


logger = logging.getLogger(__name__)


class SynchronousExecutor(cf.Executor):
    def submit(self, fn, /, *args, **kwargs):
        future = cf.Future()
        future.set_result(fn(*args, **kwargs))
        return future


def wait_on_futures(futures):
    for future in cf.as_completed(futures):
        exception = future.exception()
        if exception is not None:
            raise exception


def cancel_futures(futures):
    for future in futures:
        future.cancel()


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


# TODO: factor these functions into the BufferedArray class


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
    # TODO (maybe) add option with encoder_threads=None to run synchronously for
    # debugging using a mock Executor
    def __init__(self, buffered_arrays, encoder_threads=1):
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

    def swap_buffers(self):
        wait_on_futures(self.futures)
        self.futures = []
        for ba in self.buffered_arrays:
            self.futures.extend(
                ba.async_flush(self.executor, self.array_offset, self.next_row)
            )
            ba.swap_buffers()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Normal exit condition
            self.next_row += 1
            self.swap_buffers()
            wait_on_futures(self.futures)
        else:
            cancel_futures(self.futures)
        self.executor.shutdown()
        return False


@dataclasses.dataclass
class ProgressConfig:
    total: int = 0
    units: str = ""
    title: str = ""
    show: bool = False
    poll_interval: float = 0.001


_progress_counter = multiprocessing.Value("Q", 0)


def update_progress(inc):
    with _progress_counter.get_lock():
        _progress_counter.value += inc


def get_progress():
    with _progress_counter.get_lock():
        val = _progress_counter.value
    return val


def set_progress(value):
    with _progress_counter.get_lock():
        _progress_counter.value = value


def progress_thread_worker(config):
    pbar = tqdm.tqdm(
        total=config.total,
        desc=config.title,
        unit_scale=True,
        unit=config.units,
        smoothing=0.1,
        disable=not config.show,
    )

    while (current := get_progress()) < config.total:
        inc = current - pbar.n
        pbar.update(inc)
        time.sleep(config.poll_interval)
    pbar.close()


class ParallelWorkManager(contextlib.AbstractContextManager):
    def __init__(self, worker_processes=1, progress_config=None):
        if worker_processes <= 0:
            # NOTE: this is only for testing, not for production use!
            self.executor = SynchronousExecutor()
        else:
            self.executor = cf.ProcessPoolExecutor(
                max_workers=worker_processes,
            )
        set_progress(0)
        if progress_config is None:
            progress_config = ProgressConfig()
        self.bar_thread = threading.Thread(
            target=progress_thread_worker,
            args=(progress_config,),
            name="progress",
            daemon=True,
        )
        self.bar_thread.start()
        self.progress_config = progress_config
        self.futures = []

    def submit(self, *args, **kwargs):
        self.futures.append(self.executor.submit(*args, **kwargs))

    def results_as_completed(self):
        for future in cf.as_completed(self.futures):
            yield future.result()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            wait_on_futures(self.futures)
            set_progress(self.progress_config.total)
            timeout = None
        else:
            cancel_futures(self.futures)
            timeout = 0
        self.bar_thread.join(timeout)
        self.executor.shutdown()
        return False
