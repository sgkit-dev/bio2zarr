import numpy as np
import numpy.testing as nt
import xarray.testing as xt
import pytest
import sgkit as sg
import sgkit.io.plink
import bed_reader
import zarr

from bio2zarr import plink


class TestSmallExample:
    @pytest.fixture(scope="class")
    def bed_path(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("data")
        path = tmp_path / "example.bed"
        # 7 sites x 3 samples
        # These are counts of allele 1, so we have to flip them
        # to get the values we expect
        dosages = np.array(
            [
                [0, 1, 2],
                [1, 0, 2],
                [0, 0, 0],
                [-127, 0, 0],
                [2, 0, 0],
                [-127, -127, -127],
                [2, 2, 2],
            ],
            dtype=np.int8,
        )
        bed_reader.to_bed(path, dosages.T)
        return path

    @pytest.fixture(scope="class")
    def ds(self, tmp_path_factory, bed_path):
        tmp_path = tmp_path_factory.mktemp("data")
        zarr_path = tmp_path / "example.plink.zarr"
        plink.convert(bed_path, zarr_path)
        return sg.load_dataset(zarr_path)

    def test_genotypes(self, ds):
        call_genotype = ds.call_genotype.values
        assert call_genotype.shape == (7, 3, 2)
        nt.assert_array_equal(
            call_genotype,
            [
                [[1, 1], [1, 0], [0, 0]],
                [[1, 0], [1, 1], [0, 0]],
                [[1, 1], [1, 1], [1, 1]],
                [[-1, -1], [1, 1], [1, 1]],
                [[0, 0], [1, 1], [1, 1]],
                [[-1, -1], [-1, -1], [-1, -1]],
                [[0, 0], [0, 0], [0, 0]],
            ],
        )


class TestEqualSgkit:
    def test_simulated_example(self, tmp_path):
        data_path = "tests/data/plink/"
        bed_path = data_path + "plink_sim_10s_100v_10pmiss.bed"
        fam_path = data_path + "plink_sim_10s_100v_10pmiss.fam"
        bim_path = data_path + "plink_sim_10s_100v_10pmiss.bim"
        # print(bed_path)
        # print(fam_path)
        sg_ds = sgkit.io.plink.read_plink(
            bed_path=bed_path, fam_path=fam_path, bim_path=bim_path
        )
        out = tmp_path / "example.plink.zarr"
        plink.convert(bed_path, out)
        ds = sg.load_dataset(out)
        nt.assert_array_equal(ds.call_genotype.values, sg_ds.call_genotype.values)


class TestExample:
    """
    .bim file looks like this:

    1       1_10    0       10      A       G
    1       1_20    0       20      T       C

    Definition: https://www.cog-genomics.org/plink/1.9/formats#bim
    Chromosome code (either an integer, or 'X'/'Y'/'XY'/'MT'; '0'
        indicates unknown) or name
    Variant identifier
    Position in morgans or centimorgans (safe to use dummy value of '0')
    Base-pair coordinate (1-based; limited to 231-2)
    Allele 1 (corresponding to clear bits in .bed; usually minor)
    Allele 2 (corresponding to set bits in .bed; usually major)
    """

    @pytest.fixture(scope="class")
    def ds(self, tmp_path_factory):
        path = "tests/data/plink/example.bed"
        out = tmp_path_factory.mktemp("data") / "example.plink.zarr"
        plink.convert(path, out)
        return sg.load_dataset(out)

    def test_sample_ids(self, ds):
        nt.assert_array_equal(ds.sample_id, [f"ind{j}" for j in range(10)])

    def test_variant_position(self, ds):
        nt.assert_array_equal(ds.variant_position, [10, 20])

    def test_variant_allele(self, ds):
        nt.assert_array_equal(ds.variant_allele, [["A", "G"], ["T", "C"]])

    def test_genotypes(self, ds):
        call_genotype = ds.call_genotype.values
        assert call_genotype.shape == (2, 10, 2)
        nt.assert_array_equal(
            call_genotype,
            [
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                ],
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                ],
            ],
        )

    def test_sgkit(self, ds):
        path = "tests/data/plink/example"
        sg_ds = sgkit.io.plink.read_plink(path=path)
        # Can't compare the full dataset yet
        nt.assert_array_equal(ds.call_genotype.values, sg_ds.call_genotype.values)
        # https://github.com/pystatgen/sgkit/issues/1209
        nt.assert_array_equal(ds.variant_allele, sg_ds.variant_allele.astype(str))
        nt.assert_array_equal(ds.variant_position, sg_ds.variant_position)
        nt.assert_array_equal(ds.sample_id, sg_ds.sample_id)


class TestSimulatedExample:
    @pytest.fixture(scope="class")
    def ds(self, tmp_path_factory):
        path = "tests/data/plink/plink_sim_10s_100v_10pmiss.bed"
        out = tmp_path_factory.mktemp("data") / "example.plink.zarr"
        plink.convert(path, out)
        return sg.load_dataset(out)

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

    # @pytest.mark.xfail
    @pytest.mark.parametrize(
        ["variants_chunk_size", "samples_chunk_size"],
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
        self, ds, tmp_path, variants_chunk_size, samples_chunk_size, worker_processes
    ):
        path = "tests/data/plink/plink_sim_10s_100v_10pmiss.bed"
        out = tmp_path / "example.zarr"
        plink.convert(
            path,
            out,
            variants_chunk_size=variants_chunk_size,
            samples_chunk_size=samples_chunk_size,
            worker_processes=worker_processes,
        )
        ds2 = sg.load_dataset(out)
        xt.assert_equal(ds, ds2)
        # TODO check array chunks


@pytest.mark.parametrize(
    ["variants_chunk_size", "samples_chunk_size"],
    [
        (10, 1),
        (10, 10),
        (33, 3),
        (99, 10),
        (3, 10),
    ],
)
@pytest.mark.parametrize("worker_processes", [0])
def test_by_validating(tmp_path, variants_chunk_size, samples_chunk_size, worker_processes):
    path = "tests/data/plink/plink_sim_10s_100v_10pmiss.bed"
    out = tmp_path / "example.zarr"
    plink.convert(
        path,
        out,
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
        worker_processes=worker_processes,
    )
    plink.validate(path, out)
