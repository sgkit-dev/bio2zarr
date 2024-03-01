import numpy as np
import numpy.testing as nt
import pytest
import zarr

from bio2zarr import core


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


class TestChunkAlignedSlices:
    @pytest.mark.parametrize(
        ["n", "expected"],
        [
            (1, [(0, 20)]),
            (2, [(0, 10), (10, 20)]),
            (3, [(0, 10), (10, 15), (15, 20)]),
            (4, [(0, 5), (5, 10), (10, 15), (15, 20)]),
            (5, [(0, 5), (5, 10), (10, 15), (15, 20)]),
            (20, [(0, 5), (5, 10), (10, 15), (15, 20)]),
        ],
    )
    def test_20_chunk_5(self, n, expected):
        z = zarr.array(np.arange(20), chunks=5, dtype=int)
        result = core.chunk_aligned_slices(z, n)
        assert result == expected

    @pytest.mark.parametrize(
        ["n", "expected"],
        [
            (1, [(0, 20)]),
            (2, [(0, 14), (14, 20)]),
            (3, [(0, 7), (7, 14), (14, 20)]),
            (4, [(0, 7), (7, 14), (14, 20)]),
        ],
    )
    def test_20_chunk_7(self, n, expected):
        z = zarr.array(np.arange(20), chunks=7, dtype=int)
        result = core.chunk_aligned_slices(z, n)
        assert result == expected

    @pytest.mark.parametrize(
        ["n", "expected"],
        [
            (1, [(0, 20)]),
            (2, [(0, 20)]),
        ],
    )
    @pytest.mark.parametrize("chunks", [20, 21, 100])
    def test_20_chunk_20(self, n, expected, chunks):
        z = zarr.array(np.arange(20), chunks=chunks, dtype=int)
        result = core.chunk_aligned_slices(z, n)
        assert result == expected

    @pytest.mark.parametrize(
        ["n", "expected"],
        [
            (1, [(0, 5)]),
            (2, [(0, 3), (3, 5)]),
            (3, [(0, 2), (2, 4), (4, 5)]),
            (4, [(0, 2), (2, 3), (3, 4), (4, 5)]),
            (5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]),
            (6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]),
            (100, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]),
        ],
    )
    def test_5_chunk_1(self, n, expected):
        z = zarr.array(np.arange(5), chunks=1, dtype=int)
        result = core.chunk_aligned_slices(z, n)
        assert result == expected
