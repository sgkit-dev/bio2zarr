import numpy as np
import numpy.testing as nt
import pytest

from bio2zarr.vcf2zarr import vcz


class TestComputeLA:
    @pytest.mark.parametrize(
        ("genotypes", "expected"),
        [
            ([], []),
            ([[0, 0]], [[0, -2]]),
            ([[0, 0], [0, 0]], [[0, -2], [0, -2]]),
            ([[1, 1], [0, 0]], [[1, -2], [0, -2]]),
            ([[0, 1], [3, 2], [3, 0]], [[0, 1], [2, 3], [0, 3]]),
            ([[0, 0], [2, 3]], [[0, -2], [2, 3]]),
            ([[2, 3], [0, 0]], [[2, 3], [0, -2]]),
            ([[128, 0], [6, 5]], [[0, 128], [5, 6]]),
            ([[0, -1], [-1, 5]], [[0, -2], [5, -2]]),
            ([[-1, -1], [-1, 5]], [[-2, -2], [5, -2]]),
        ],
    )
    def test_simple_examples(self, genotypes, expected):
        G = np.array(genotypes)
        result = vcz.compute_la_field(G)
        nt.assert_array_equal(result, expected)

    def test_extreme_value(self):
        G = np.array([[0, 2**32 - 1]])
        with pytest.raises(ValueError, match="Extreme"):
            vcz.compute_la_field(G)


class TestComputeLAD:
    @pytest.mark.parametrize(
        ("ad", "la", "expected"),
        [
            # Missing data
            ([[0, 0]], [[-2, -2]], [[-2, -2]]),
            # 0/0 calls
            ([[10, 0]], [[0, -2]], [[10, -2]]),
            ([[10, 0, 0]], [[0, -2]], [[10, -2]]),
            ([[10, 0, 0], [11, 0, 0]], [[0, -2], [0, -2]], [[10, -2], [11, -2]]),
            # 0/1 calls
            ([[10, 11]], [[0, 1]], [[10, 11]]),
            ([[10, 11], [12, 0]], [[0, 1], [0, -2]], [[10, 11], [12, -2]]),
            # 0/2 calls
            ([[10, 0, 11]], [[0, 2]], [[10, 11]]),
            ([[10, 0, 11], [10, 11, 0]], [[0, 2], [0, 1]], [[10, 11], [10, 11]]),
            (
                [[10, 0, 11], [10, 11, 0], [12, 0, 0]],
                [[0, 2], [0, 1], [0, -2]],
                [[10, 11], [10, 11], [12, -2]],
            ),
            # 1/2 calls
            ([[0, 10, 11]], [[1, 2]], [[10, 11]]),
            ([[0, 10, 11], [12, 0, 13]], [[1, 2], [0, 2]], [[10, 11], [12, 13]]),
            (
                [[0, 10, 11], [12, 0, 13], [14, 0, 0]],
                [[1, 2], [0, 2], [0, -2]],
                [[10, 11], [12, 13], [14, -2]],
            ),
        ],
    )
    def test_simple_examples(self, ad, la, expected):
        result = vcz.compute_lad_field(np.array(ad), np.array(la))
        nt.assert_array_equal(result, expected)


# PL translation indexes:
# a       b       i
# 0       0       0
# 0       1       1
# 0       2       3
# 0       3       6
# 1       1       2
# 1       2       4
# 1       3       7
# 2       2       5
# 2       3       8
# 3       3       9


class TestComputeLPL:
    @pytest.mark.parametrize(
        ("pl", "la", "expected"),
        [
            # Missing
            ([range(3)], [[-2, -2]], [[-2, -2, -2]]),
            # 0/0 calls
            ([range(3)], [[0, -2]], [[0, -2, -2]]),
            # 0/0 calls
            ([[-1, -1, -1]], [[0, -2]], [[-1, -2, -2]]),
            # 1/1 calls
            ([range(3)], [[1, -2]], [[2, -2, -2]]),
            ([range(3), range(3)], [[0, -2], [1, -2]], [[0, -2, -2], [2, -2, -2]]),
            # 2/2 calls
            ([range(6)], [[2, -2]], [[5, -2, -2]]),
            # 0/1 calls
            ([range(3)], [[0, 1]], [[0, 1, 2]]),
            # 0/2 calls
            ([range(6)], [[0, 2]], [[0, 3, 5]]),
        ],
    )
    def test_simple_examples(self, pl, la, expected):
        result = vcz.compute_lpl_field(np.array(pl), np.array(la))
        nt.assert_array_equal(result, expected)
