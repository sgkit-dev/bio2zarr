import numpy as np
import numpy.testing as nt
import pytest
import zarr

from bio2zarr import core


def encode_arrays(arrays, data, encoder_threads=1):
    buffered_arrays = [core.BufferedArray(a) for a in arrays]
    assert len(arrays) == len(data)
    for a, d in zip(arrays, data):
        assert a.shape == d.shape
        assert a.shape[0] == arrays[0].shape[0]
    data_row = 0
    with core.ThreadedZarrEncoder(buffered_arrays, encoder_threads) as tze:
        for data_row in range(len(data[0])):
            j = tze.next_buffer_row()
            for ba, data_array in zip(buffered_arrays, data):
                ba.buff[j] = data_array[data_row]


class TestZarrEncoder:
    @pytest.mark.parametrize(
        ["data", "chunk_size"],
        [
            (np.arange(10), (1,)),
            (np.arange(10), (3,)),
            (np.arange(10), (5,)),
            (np.arange(10), (10,)),
            (np.arange(10, dtype=np.int8), (3,)),
            (np.arange(10, dtype=np.int32), (3,)),
            (np.arange(10, dtype=np.float32), (3,)),
            (np.arange(10, dtype=np.float64), (3,)),
            (-1 * np.arange(100, dtype=np.int32)[::-1], (7,)),
            # 2D arrays
            (np.arange(16).reshape((4, 4)), (1, 4)),
            (np.arange(16).reshape((4, 4)), (3, 3)),
            (np.arange(16).reshape((4, 4)), (16, 1)),
            # 3D arrays
            (np.arange(32).reshape((8, 2, 2)), (1, 4, 2)),
        ],
    )
    def test_single_array(self, data, chunk_size):
        a = zarr.empty_like(data, chunks=chunk_size)
        encode_arrays([a], [data])
        nt.assert_array_equal(a[:], data)

    @pytest.mark.parametrize("chunk_size", range(1, 6))
    def test_multi_array(self, chunk_size):
        n = 33
        data = [
            np.arange(n),
            np.arange(n, dtype=np.int32),
            np.arange(n, dtype=np.float64),
        ]
        arrays = [zarr.empty_like(d, chunks=(chunk_size,)) for d in data]
        encode_arrays(arrays, data)

    @pytest.mark.parametrize("threads", range(1, 6))
    def test_single_array_threads(self, threads):
        data = np.arange(10_333)
        a = zarr.empty_like(data, chunks=(100,))
        encode_arrays([a], [data], threads)
        nt.assert_array_equal(a[:], data)

    def test_error_in_user_code(self):
        data = list(range(10)) + ["string"]
        a = zarr.empty(len(data), chunks=(1,), dtype=int)
        ba = core.BufferedArray(a)

        with pytest.raises(ValueError, match="int()"):
            with core.ThreadedZarrEncoder([ba]) as tze:
                for d in data:
                    j = tze.next_buffer_row()
                    # This raises an error when "string" inserted to buffer
                    ba.buff[j] = d

    def test_error_in_encode(self):
        data = np.array([1])
        a = zarr.empty_like(data, chunks=(1,))
        ba = core.BufferedArray(a)

        with pytest.raises(ValueError, match="int()"):
            with core.ThreadedZarrEncoder([ba]) as tze:
                for d in data:
                    j = tze.next_buffer_row()
                    # This raises an error when "string" inserted to buffer
                    ba.buff[j] = d
                # We only flush on exiting the context manager, so switch the
                # buffer for something nasty.
                # NB: this is the only reliable way I can think of raising
                # an error in the futures. In reality these will happen
                # when we run out of disk space, but this is hard to simulate
                ba.buff = np.array(["not an integer"])


class TestParallelWorkManager:
    @pytest.mark.parametrize("total", [1, 10, 2**63])
    @pytest.mark.parametrize("workers", [0, 1])
    def test_one_future_progress(self, total, workers):
        progress_config = core.ProgressConfig(total=total)
        with core.ParallelWorkManager(workers, progress_config) as pwm:
            pwm.submit(core.update_progress, total)
        assert core.get_progress() == total

    @pytest.mark.parametrize("total", [1, 10, 1000])
    @pytest.mark.parametrize("workers", [0, 1, 2, 3])
    def test_n_futures_progress(self, total, workers):
        progress_config = core.ProgressConfig(total=total)
        with core.ParallelWorkManager(workers, progress_config) as pwm:
            for _ in range(total):
                pwm.submit(core.update_progress, 1)
        assert core.get_progress() == total

    @pytest.mark.parametrize("total", [1, 10, 20])
    @pytest.mark.parametrize("workers", [0, 1, 2, 3])
    def test_results_as_completed(self, total, workers):
        with core.ParallelWorkManager(workers) as pwm:
            for j in range(total):
                pwm.submit(frozenset, range(j))
            results = set(pwm.results_as_completed())
            assert results == set(frozenset(range(j)) for j in range(total))

    @pytest.mark.parametrize("total", [1, 10, 20])
    @pytest.mark.parametrize("workers", [1, 2, 3])
    def test_error_in_workers_as_completed(self, total, workers):
        with pytest.raises(TypeError):
            with core.ParallelWorkManager(workers) as pwm:
                for j in range(total):
                    pwm.submit(frozenset, range(j))
                # Raises a TypeError:
                pwm.submit(frozenset, j)
                set(pwm.results_as_completed())

    @pytest.mark.parametrize("total", [1, 10, 20])
    @pytest.mark.parametrize("workers", [1, 2, 3])
    def test_error_in_workers_on_exit(self, total, workers):
        with pytest.raises(TypeError):
            with core.ParallelWorkManager(workers) as pwm:
                for j in range(total):
                    pwm.submit(frozenset, range(j))
                # Raises a TypeError:
                pwm.submit(frozenset, j)
