import numpy as np
import pytest
import zarr

from bio2zarr import core


class TestMinIntDtype:
    @pytest.mark.parametrize(
        ("min_value", "max_value", "dtype"),
        [
            (0, 1, "i1"),
            (0, 0, "i1"),
            (0, 127, "i1"),
            (127, 128, "i2"),
            (-127, 0, "i1"),
            (-127, -126, "i1"),
            (0, 2**15 - 1, "i2"),
            (-(2**15), 2**15 - 1, "i2"),
            (0, 2**15, "i4"),
            (-(2**15), 2**15, "i4"),
            (0, 2**31 - 1, "i4"),
            (-(2**31), 2**31 - 1, "i4"),
            (2**31 - 1, 2**31 - 1, "i4"),
            (0, 2**31, "i8"),
            (0, 2**32, "i8"),
        ],
    )
    def test_values(self, min_value, max_value, dtype):
        assert core.min_int_dtype(min_value, max_value) == dtype

    @pytest.mark.parametrize(
        ("min_value", "max_value"),
        [
            (0, 2**63),
            (-(2**63) - 1, 0),
            (0, 2**65),
        ],
    )
    def test_overflow(self, min_value, max_value):
        with pytest.raises(OverflowError, match="Integer cannot"):
            core.min_int_dtype(min_value, max_value)

    @pytest.mark.parametrize(
        ("min_value", "max_value"),
        [
            (1, 0),
            (-1, -2),
            (2**31, 2**31 - 1),
        ],
    )
    def test_bad_min_max(self, min_value, max_value):
        with pytest.raises(ValueError, match="must be <="):
            core.min_int_dtype(min_value, max_value)


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
        with pytest.raises(TypeError):  # noqa PT012
            with core.ParallelWorkManager(workers) as pwm:
                for j in range(total):
                    pwm.submit(frozenset, range(j))
                # Raises a TypeError:
                pwm.submit(frozenset, j)
                set(pwm.results_as_completed())

    @pytest.mark.parametrize("total", [1, 10, 20])
    @pytest.mark.parametrize("workers", [1, 2, 3])
    def test_error_in_workers_on_exit(self, total, workers):
        with pytest.raises(TypeError):  # noqa PT012
            with core.ParallelWorkManager(workers) as pwm:
                for j in range(total):
                    pwm.submit(frozenset, range(j))
                # Raises a TypeError:
                pwm.submit(frozenset, j)


class TestChunkAlignedSlices:
    @pytest.mark.parametrize(
        ("n", "expected"),
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
        ("n", "max_chunks", "expected"),
        [
            (1, 5, [(0, 20)]),
            (1, 1, [(0, 5)]),
            (2, 1, [(0, 5)]),
            (3, 1, [(0, 5)]),
            (2, 3, [(0, 10), (10, 15)]),
            (2, 4, [(0, 10), (10, 20)]),
        ],
    )
    def test_20_chunk_5_max_chunks(self, n, max_chunks, expected):
        z = zarr.array(np.arange(20), chunks=5, dtype=int)
        result = core.chunk_aligned_slices(z, n, max_chunks=max_chunks)
        assert result == expected

    @pytest.mark.parametrize(
        ("n", "expected"),
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
        ("n", "expected"),
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
        ("n", "expected"),
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


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        # NOTE: this data was generated using du -sb on a Linux system.
        # It *might* work in CI, but it may well not either, as it's
        # probably dependent on a whole bunch of things. Expect to fail
        # at some point.
        ("tests/data", 4630726),
        ("tests/data/vcf", 4618589),
        ("tests/data/vcf/sample.vcf.gz", 1089),
    ],
)
def test_du(path, expected):
    assert core.du(path) == expected
