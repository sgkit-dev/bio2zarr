import numpy as np
import numpy.testing as nt
import pytest

from bio2zarr.vcf2zarr.vcz import compute_laa_field, compute_lad_field


class TestComputeLAA:
    @pytest.mark.parametrize(
        ("genotypes", "expected"),
        [
            ([], []),
            ([[0, 0]], [[-2, -2]]),
            ([[0, 0], [0, 0]], [[-2, -2], [-2, -2]]),
            ([[1, 1], [0, 0]], [[1, -2], [-2, -2]]),
            ([[0, 1], [3, 2], [3, 0]], [[1, -2], [2, 3], [3, -2]]),
            ([[0, 0], [2, 3]], [[-2, -2], [2, 3]]),
            ([[2, 3], [0, 0]], [[2, 3], [-2, -2]]),
            ([[128, 0], [6, 5]], [[128, -2], [5, 6]]),
            ([[0, -1], [-1, 5]], [[-2, -2], [5, -2]]),
        ],
    )
    def test_simple_examples(self, genotypes, expected):
        G = np.array(genotypes)
        result = compute_laa_field(G)
        nt.assert_array_equal(result, expected)

    def test_extreme_value(self):
        G = np.array([[0, 2**32 - 1]])
        with pytest.raises(ValueError, match="Extreme"):
            compute_laa_field(G)


class TestComputeLAD:
    @pytest.mark.parametrize(
        ("ad", "laa", "expected"),
        [
            # 0/0 calls
            ([[10, 0]], [[-2, -2]], [[10, -2]]),
            ([[10, 0, 0], [11, 0, 0]], [[-2, -2], [-2, -2]], [[10, -2], [11, -2]]),
            # 0/1 calls
            ([[10, 11]], [[1, -2]], [[10, 11]]),
            ([[10, 11], [12, 0]], [[1, -2], [-2, -2]], [[10, 11], [12, -2]]),
            # 0/2 calls
            ([[10, 0, 11]], [[2, -2]], [[10, 11]]),
            ([[10, 0, 11], [10, 11, 0]], [[2, -2], [1, -2]], [[10, 11], [10, 11]]),
            (
                [[10, 0, 11], [10, 11, 0], [12, 0, 0]],
                [[2, -2], [1, -2], [-2, -2]],
                [[10, 11], [10, 11], [12, -2]],
            ),
            # 1/2 calls
            ([[0, 10, 11]], [[1, 2]], [[10, 11]]),
            ([[0, 10, 11], [12, 0, 13]], [[1, 2], [2, -2]], [[10, 11], [12, 13]]),
            (
                [[0, 10, 11], [12, 0, 13], [14, 0, 0]],
                [[1, 2], [2, -2], [-2, -2]],
                [[10, 11], [12, 13], [14, -2]],
            ),
        ],
    )
    def test_simple_examples(self, ad, laa, expected):
        result = compute_lad_field(np.array(ad), np.array(laa))
        nt.assert_array_equal(result, expected)
