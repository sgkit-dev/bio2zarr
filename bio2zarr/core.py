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
import numcodecs


logger = logging.getLogger(__name__)

numcodecs.blosc.use_threads = False

# TODO this should probably go in another module where we abstract
# out the zarr defaults
default_compressor = numcodecs.Blosc(
    cname="zstd", clevel=7, shuffle=numcodecs.Blosc.AUTOSHUFFLE
)


def chunk_aligned_slices(z, n):
    """
    Returns at n slices in the specified zarr array, aligned
    with its chunks
    """
    chunk_size = z.chunks[0]
    num_chunks = int(np.ceil(z.shape[0] / chunk_size))
    slices = []
    splits = np.array_split(np.arange(num_chunks), min(n, num_chunks))
    for split in splits:
        start = split[0] * chunk_size
        stop = (split[-1] + 1) * chunk_size
        stop = min(stop, z.shape[0])
        slices.append((start, stop))
    return slices


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
    array_offset: int
    buff: np.ndarray
    buffer_row: int

    def __init__(self, array, offset):
        self.array = array
        self.array_offset = offset
        assert offset % array.chunks[0] == 0
        dims = list(array.shape)
        dims[0] = min(array.chunks[0], array.shape[0])
        self.buff = np.zeros(dims, dtype=array.dtype)
        self.buffer_row = 0

    @property
    def chunk_length(self):
        return self.buff.shape[0]

    def next_buffer_row(self):
        if self.buffer_row == self.chunk_length:
            self.flush()
        row = self.buffer_row
        self.buffer_row += 1
        return row

    def flush(self):
        # TODO just move sync_flush_array in here
        if self.buffer_row != 0:
            if len(self.array.chunks) <= 1:
                sync_flush_1d_array(
                    self.buff[: self.buffer_row], self.array, self.array_offset
                )
            else:
                sync_flush_2d_array(
                    self.buff[: self.buffer_row], self.array, self.array_offset
                )
            logger.debug(
                f"Flushed chunk {self.array} {self.array_offset} + {self.buffer_row}")
            self.array_offset += self.chunk_length
            self.buffer_row = 0


def sync_flush_1d_array(np_buffer, zarr_array, offset):
    zarr_array[offset : offset + np_buffer.shape[0]] = np_buffer
    update_progress(1)


def sync_flush_2d_array(np_buffer, zarr_array, offset):
    # Write chunks in the second dimension 1-by-1 to make progress more
    # incremental, and to avoid large memcopies in the underlying
    # encoder implementations.
    s = slice(offset, offset + np_buffer.shape[0])
    chunk_width = zarr_array.chunks[1]
    zarr_array_width = zarr_array.shape[1]
    start = 0
    while start < zarr_array_width:
        stop = min(start + chunk_width, zarr_array_width)
        zarr_array[s, start:stop] = np_buffer[:, start:stop]
        update_progress(1)
        start = stop


@dataclasses.dataclass
class ProgressConfig:
    total: int = 0
    units: str = ""
    title: str = ""
    show: bool = False
    poll_interval: float = 0.001


# NOTE: this approach means that we cannot have more than one
# progressable thing happening per source process. This is
# probably fine in practise, but there could be corner cases
# where it's not. Something to watch out for.
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
        desc=f"{config.title:>7}",
        unit_scale=True,
        unit=config.units,
        smoothing=0.1,
        disable=not config.show,
    )

    while (current := get_progress()) < config.total:
        inc = current - pbar.n
        pbar.update(inc)
        time.sleep(config.poll_interval)
    # TODO figure out why we're sometimes going over total
    # if get_progress() != config.total:
    #     print("HOW DID THIS HAPPEN!!")
    #     print(get_progress())
    #     print(config)
    # assert get_progress() == config.total
    inc = config.total - pbar.n
    pbar.update(inc)
    pbar.close()
    # print("EXITING PROGRESS THREAD")


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
            # Note: this doesn't seem to be working correctly. If
            # we set a timeout of None we get deadlocks
            set_progress(self.progress_config.total)
            timeout = None
        else:
            cancel_futures(self.futures)
            timeout = 0
        self.bar_thread.join(timeout)
        self.executor.shutdown()
        return False
