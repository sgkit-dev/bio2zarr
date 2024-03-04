import dataclasses
import contextlib
import concurrent.futures as cf
import multiprocessing
import threading
import logging
import time

import zarr
import numpy as np
# import tqdm
import rich.progress as rp
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


class ParallelWorkManager(contextlib.AbstractContextManager):
    def __init__(self, worker_processes=1, progress_config=None):
        if worker_processes <= 0:
            # NOTE: this is only for testing, not for production use!
            self.executor = SynchronousExecutor()
        else:
            self.executor = cf.ProcessPoolExecutor(
                max_workers=worker_processes,
            )
        self.futures = []

        set_progress(0)
        if progress_config is None:
            progress_config = ProgressConfig()
        self.progress_config = progress_config
        # self.progress_bar = tqdm.tqdm(
        #     total=progress_config.total,
        #     desc=f"{progress_config.title:>7}",
        #     unit_scale=True,
        #     unit=progress_config.units,
        #     smoothing=0.1,
        #     disable=not progress_config.show,
        # )
        self.progress_bar = rp.Progress(
            rp.TimeElapsedColumn(),
            rp.TextColumn("[progress.description]{task.description}"),
            rp.BarColumn(),
            rp.TaskProgressColumn(),
            rp.MofNCompleteColumn(),
            rp.TextColumn(f"{progress_config.units} ETA:"),
            rp.TimeRemainingColumn())
        self.progress_task = self.progress_bar.add_task(
            f"{progress_config.title:>7}",
            total=progress_config.total)
        self.progress_bar.start()
        self.completed = False
        self.completed_lock = threading.Lock()
        self.progress_thread = threading.Thread(
            target=self._update_progress_worker,
            name="progress-update",
        )
        self.progress_thread.start()

    def _update_progress(self):
        current = get_progress()
        # inc = current - self.progress_bar.n
        # print("UPDATE PROGRESS: current = ", current)
        # self.progress_bar.update(inc)
        self.progress_bar.update(self.progress_task, completed=current, refresh=True)

    def _update_progress_worker(self):
        completed = False
        while not completed:
            self._update_progress()
            time.sleep(self.progress_config.poll_interval)
            with self.completed_lock:
                completed = self.completed
        logger.debug("Exit progress thread")

    def submit(self, *args, **kwargs):
        self.futures.append(self.executor.submit(*args, **kwargs))

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
        self.progress_bar.stop()
        return False
