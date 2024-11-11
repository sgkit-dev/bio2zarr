import concurrent.futures as cf
import contextlib
import dataclasses
import json
import logging
import math
import multiprocessing
import os
import os.path
import threading
import time

import humanfriendly
import numcodecs
import numpy as np
import tqdm
import zarr

logger = logging.getLogger(__name__)

numcodecs.blosc.use_threads = False


def display_number(x):
    ret = "n/a"
    if math.isfinite(x):
        ret = f"{x: 0.2g}"
    return ret


def display_size(n):
    return humanfriendly.format_size(n, binary=True)


def min_int_dtype(min_value, max_value):
    if min_value > max_value:
        raise ValueError("min_value must be <= max_value")
    for a_dtype in ["i1", "i2", "i4", "i8"]:
        info = np.iinfo(a_dtype)
        if info.min <= min_value and max_value <= info.max:
            return a_dtype
    raise OverflowError("Integer cannot be represented")


def chunk_aligned_slices(z, n, max_chunks=None):
    """
    Returns at n slices in the specified zarr array, aligned
    with its chunks
    """
    chunk_size = z.chunks[0]
    num_chunks = int(np.ceil(z.shape[0] / chunk_size))
    if max_chunks is not None:
        num_chunks = min(num_chunks, max_chunks)
    slices = []
    splits = np.array_split(np.arange(num_chunks), min(n, num_chunks))
    for split in splits:
        start = split[0] * chunk_size
        stop = (split[-1] + 1) * chunk_size
        stop = min(stop, z.shape[0])
        slices.append((start, stop))
    return slices


def du(path):
    """
    Return the total bytes stored at this path.
    """
    total = os.path.getsize(path)
    # pathlib walk method doesn't exist until 3.12 :(
    for root, dirs, files in os.walk(path):
        for lst in [dirs, files]:
            for name in lst:
                fullname = os.path.join(root, name)
                size = os.path.getsize(fullname)
                total += size
    logger.debug(f"du({path}) = {total}")
    return total


class SynchronousExecutor(cf.Executor):
    # Arguably we should use workers=0 as the default and use this
    # executor implementation. However, the docs are fairly explicit
    # about saying we shouldn't instantiate Future objects directly,
    # so it's best to keep this as a semi-secret debugging interface
    # for now.
    def submit(self, fn, /, *args, **kwargs):
        future = cf.Future()
        future.set_result(fn(*args, **kwargs))
        return future


def wait_on_futures(futures):
    for future in cf.as_completed(futures):
        exception = future.exception()
        if exception is not None:
            cancel_futures(futures)
            if isinstance(exception, cf.process.BrokenProcessPool):
                raise RuntimeError(
                    "Worker process died: you may have run out of memory"
                ) from exception
            else:
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
        self.buff = np.empty(dims, dtype=array.dtype)
        # Explicitly Fill with zeros here to make any out-of-memory errors happen
        # quickly.
        self.buff[:] = 0
        self.buffer_row = 0

    @property
    def variants_chunk_size(self):
        return self.buff.shape[0]

    def next_buffer_row(self):
        if self.buffer_row == self.variants_chunk_size:
            self.flush()
        row = self.buffer_row
        self.buffer_row += 1
        return row

    def flush(self):
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
                f"Flushed <{self.array.name} {self.array.shape} "
                f"{self.array.dtype}> "
                f"{self.array_offset}:{self.array_offset + self.buffer_row}"
                f"{self.buff.nbytes / 2**20: .2f}Mb"
            )
            self.array_offset += self.variants_chunk_size
            self.buffer_row = 0


def sync_flush_1d_array(np_buffer, zarr_array, offset):
    zarr_array[offset : offset + np_buffer.shape[0]] = np_buffer
    update_progress(np_buffer.nbytes)


def sync_flush_2d_array(np_buffer, zarr_array, offset):
    # Write chunks in the second dimension 1-by-1 to make progress more
    # incremental, and to avoid large memcopies in the underlying
    # encoder implementations.
    s = slice(offset, offset + np_buffer.shape[0])
    samples_chunk_size = zarr_array.chunks[1]
    # TODO use zarr chunks here for simplicity
    zarr_array_width = zarr_array.shape[1]
    start = 0
    while start < zarr_array_width:
        stop = min(start + samples_chunk_size, zarr_array_width)
        chunk_buffer = np_buffer[:, start:stop]
        zarr_array[s, start:stop] = chunk_buffer
        update_progress(chunk_buffer.nbytes)
        start = stop


@dataclasses.dataclass
class ProgressConfig:
    total: int = 0
    units: str = ""
    title: str = ""
    show: bool = False
    poll_interval: float = 0.01


# NOTE: this approach means that we cannot have more than one
# progressable thing happening per source process. This is
# probably fine in practise, but there could be corner cases
# where it's not. Something to watch out for.
_progress_counter = None


def update_progress(inc):
    # If the _progress_counter has not been set we are working in a
    # synchronous non-progress tracking context
    if _progress_counter is not None:
        with _progress_counter.get_lock():
            _progress_counter.value += inc


def get_progress():
    with _progress_counter.get_lock():
        val = _progress_counter.value
    return val


def setup_progress_counter(counter):
    global _progress_counter
    _progress_counter = counter


class ParallelWorkManager(contextlib.AbstractContextManager):
    def __init__(self, worker_processes=1, progress_config=None):
        # Need to specify this explicitly to suppport Macs and
        # for future proofing.
        ctx = multiprocessing.get_context("spawn")
        global _progress_counter
        _progress_counter = ctx.Value("Q", 0)
        if worker_processes <= 0:
            # NOTE: this is only for testing and debugging, not for
            # production. See note on the SynchronousExecutor class.
            self.executor = SynchronousExecutor()
        else:
            self.executor = cf.ProcessPoolExecutor(
                max_workers=worker_processes,
                mp_context=ctx,
                initializer=setup_progress_counter,
                initargs=(_progress_counter,),
            )
        self.futures = set()

        if progress_config is None:
            progress_config = ProgressConfig()
        self.progress_config = progress_config
        self.progress_bar = tqdm.tqdm(
            total=progress_config.total,
            desc=f"{progress_config.title:>8}",
            unit_scale=True,
            unit=progress_config.units,
            smoothing=0.1,
            disable=not progress_config.show,
        )
        self.completed = False
        self.completed_lock = threading.Lock()
        self.progress_thread = threading.Thread(
            target=self._update_progress_worker,
            name="progress-update",
            daemon=True,  # Avoids deadlock on exit in awkward error conditions
        )
        self.progress_thread.start()

    def _update_progress(self):
        current = get_progress()
        inc = current - self.progress_bar.n
        self.progress_bar.update(inc)

    def _update_progress_worker(self):
        completed = False
        while not completed:
            self._update_progress()
            time.sleep(self.progress_config.poll_interval)
            with self.completed_lock:
                completed = self.completed
        logger.debug("Exit progress thread")

    def submit(self, *args, **kwargs):
        future = self.executor.submit(*args, **kwargs)
        self.futures.add(future)
        return future

    def results_as_completed(self):
        for future in cf.as_completed(self.futures):
            yield future.result()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            wait_on_futures(self.futures)
        else:
            cancel_futures(self.futures)
        # There's probably a much cleaner way of doing this with a Condition
        # or something, but this seems to work OK for now. This setup might
        # make small conversions a bit laggy as we wait on the sleep interval
        # though.
        with self.completed_lock:
            self.completed = True
        self.executor.shutdown(wait=False)
        # FIXME there's currently some thing weird happening at the end of
        # Encode 1D for 1kg-p3. The progress bar disappears, like we're
        # setting a total of zero or something.
        self.progress_thread.join()
        self._update_progress()
        self.progress_bar.close()
        return False


class JsonDataclass:
    def asdict(self):
        return dataclasses.asdict(self)

    def asjson(self):
        return json.dumps(self.asdict(), indent=4)
