import numpy as np
import numpy.testing as nt
import xarray.testing as xt
import pytest
import sgkit as sg
import zarr

from bio2zarr import plink


class TestSmallExample:
    @pytest.fixture(scope="class")
    def ds(self, tmp_path_factory):
        path = "tests/data/plink/plink_sim_10s_100v_10pmiss.bed"
        out = tmp_path_factory.mktemp("data") / "example.vcf.zarr"
        plink.convert(path, out)
        return sg.load_dataset(out)


    @pytest.mark.xfail
    def test_genotypes(self, ds):
        # Validate a few randomly selected individual calls
        # (spanning all possible states for a call)
        idx = np.array(
            [
                [50, 7],
                [81, 8],
                [45, 2],
                [36, 8],
                [24, 2],
                [92, 9],
                [26, 2],
                [81, 0],
                [31, 8],
                [4, 9],
            ]
        )
        expected = np.array(
            [
                [1, 0],
                [1, 0],
                [0, 0],
                [0, 0],
                [-1, -1],
                [1, 1],
                [1, 1],
                [0, 0],
                [1, 1],
                [1, 1],
            ]
        )
        gt = ds["call_genotype"].values
        actual = gt[tuple(idx.T)]
        # print(actual)
        # print(expected)
        # FIXME not working
        nt.assert_array_equal(actual, expected)

    @pytest.mark.xfail
    @pytest.mark.parametrize(
        ["chunk_length", "chunk_width"],
        [
            (10, 1),
            (10, 10),
            (33, 3),
            # This one doesn't fail as it's the same as defaults
            # (100, 10),
        ],
    )
    @pytest.mark.parametrize("worker_processes", [0, 1, 2])
    def test_chunk_size(
        self, ds, tmp_path, chunk_length, chunk_width, worker_processes
    ):
        path = "tests/data/plink/plink_sim_10s_100v_10pmiss.bed"
        out = tmp_path / "example.zarr"
        plink.convert(path, out, chunk_length=chunk_length, chunk_width=chunk_width,
                worker_processes=worker_processes)
        ds2 = sg.load_dataset(out)
        # print(ds2)
        # print(ds2.call_genotype.values)
        # print(ds.call_genotype.values)
        xt.assert_equal(ds, ds2)
        # TODO check array chunks
