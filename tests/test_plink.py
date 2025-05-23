import os.path

import bed_reader
import numpy as np
import numpy.testing as nt
import pytest
import sgkit as sg
import xarray.testing as xt
import zarr

from bio2zarr import plink, vcf


class TestReadBim:
    def test_example(self):
        path = "tests/data/plink/example.bim"
        df = plink.read_bim(path)
        # chrom, vid, cm_pos, pos, a1, a2
        # 1       1_10    0       10      A       GG
        # 1       1_20    0       20      TTT     C
        nt.assert_array_equal(df["contig"].values, ["1", "1"])
        nt.assert_array_equal(df["variant_id"].values, ["1_10", "1_20"])
        nt.assert_array_equal(df["cm_position"].values, [0, 0])
        nt.assert_array_equal(df["position"].values, [10, 20])
        nt.assert_array_equal(df["allele_1"].values, ["A", "TTT"])
        nt.assert_array_equal(df["allele_2"].values, ["GG", "C"])


class TestReadFam:
    def test_example(self):
        path = "tests/data/plink/example.fam"
        df = plink.read_fam(path)
        # FID IID FATHER MOTHER SEX PHENOTYPE
        # ind0 ind0 0 0 0 -9
        # ind1 ind1 0 0 0 -9
        # ind2 ind2 0 0 0 -9
        # ind3 ind3 0 0 0 -9
        # ind4 ind4 0 0 0 -9
        # ind5 ind5 0 0 0 -9
        # ind6 ind6 0 0 0 -9
        # ind7 ind7 0 0 0 -9
        # ind8 ind8 0 0 0 -9
        # ind9 ind9 0 0 0 -9
        nt.assert_array_equal(df["family_id"].values, [f"ind{j}" for j in range(10)])
        nt.assert_array_equal(
            df["individual_id"].values, [f"ind{j}" for j in range(10)]
        )
        nt.assert_array_equal(df["paternal_id"].values, ["0" for j in range(10)])
        nt.assert_array_equal(df["maternal_id"].values, ["0" for j in range(10)])
        nt.assert_array_equal(df["sex"].values, ["0" for j in range(10)])
        nt.assert_array_equal(df["phenotype"].values, ["-9" for j in range(10)])


class TestBedReader:
    @pytest.mark.parametrize(
        "path",
        [
            "tests/data/plink/example.bim",
            "tests/data/vcf/sample.vcf.gz",
        ],
    )
    def test_bad_file_type(self, path):
        with pytest.raises(ValueError, match="Invalid BED file magic bytes"):
            plink.BedReader(path, 1, 1)

    @pytest.mark.parametrize(
        ("num_variants", "num_samples"),
        [
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (1, 8),
            (1, 9),
            (2, 1),
            (3, 1),
            (10, 1),
            (100, 1),
            (10, 2),
            (10, 3),
            (10, 4),
            (10, 5),
            (20, 20),
            (30, 3),
        ],
    )
    def test_generated_bed_files(self, tmp_path, num_variants, num_samples):
        bed_file = tmp_path / "a_file.bed"
        # Generate a regular pattern of all possible values
        data = np.arange(num_variants * num_samples, dtype=int) % 4
        data[data == 3] = -127
        data = data.reshape((num_variants, num_samples))
        bed_reader.to_bed(bed_file, data.T, num_threads=1)

        bytes_per_variant = (num_samples + 3) // 4
        expected_size = 3 + bytes_per_variant * num_variants
        assert os.path.getsize(bed_file) == expected_size

        br_map = {0: (0, 0), 1: (0, 1), 2: (1, 1), -127: (-1, -1)}
        reader = plink.BedReader(bed_file, num_variants, num_samples)
        g = reader.decode(0, num_variants)
        assert g.shape == (num_variants, num_samples, 2)
        for j in range(num_variants):
            for k in range(num_samples):
                assert br_map[data[j, k]] == tuple(g[j, k])

    @pytest.mark.parametrize(
        ("num_variants", "num_samples"),
        [
            (1, 1),
            (30, 3),
            (300, 1000),
        ],
    )
    @pytest.mark.parametrize("buffer_size", [0, 1, 3, 100, 100_000, None])
    def test_iter_decode(self, tmp_path, buffer_size, num_variants, num_samples):
        bed_file = tmp_path / "a_file.bed"
        # Generate a regular pattern of all possible values
        data = np.arange(num_variants * num_samples, dtype=int) % 4
        data[data == 3] = -127
        data = data.reshape((num_variants, num_samples))
        bed_reader.to_bed(bed_file, data.T, num_threads=1)

        reader = plink.BedReader(bed_file, num_variants, num_samples)
        G = reader.decode(0, num_variants)
        assert G.shape == (num_variants, num_samples, 2)

        reader = plink.BedReader(bed_file, num_variants, num_samples)
        G2 = list(reader.iter_decode(0, num_variants, buffer_size=buffer_size))
        nt.assert_array_equal(G, G2)


class TestSmallExample:
    @pytest.fixture(scope="class")
    def bed_path(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("data")
        path = tmp_path / "example.bed"
        # 7 sites x 3 samples
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
        m = 7
        d = {
            "chromosome": ["chr1"] * m,
            "sid": [f"id{j}" for j in range(m)],
            "bp_position": range(1, m + 1),
            "allele_1": ["A"] * m,
            "allele_2": ["T"] * m,
            "iid": [f"s{j}" for j in range(3)],
        }
        bed_reader.to_bed(path, dosages.T, properties=d)
        return path.with_suffix("")

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
                [[0, 0], [0, 1], [1, 1]],
                [[0, 1], [0, 0], [1, 1]],
                [[0, 0], [0, 0], [0, 0]],
                [[-1, -1], [0, 0], [0, 0]],
                [[1, 1], [0, 0], [0, 0]],
                [[-1, -1], [-1, -1], [-1, -1]],
                [[1, 1], [1, 1], [1, 1]],
            ],
        )

    def test_variant_position(self, ds):
        nt.assert_array_equal(ds["variant_position"], np.arange(1, 8))

    def test_variant_contig(self, ds):
        nt.assert_array_equal(ds["variant_contig"], np.zeros(7))
        nt.assert_array_equal(ds["contig_id"], ["chr1"])

    def test_variant_id(self, ds):
        nt.assert_array_equal(ds["variant_id"], [f"id{j}" for j in range(7)])

    def test_variant_allele(self, ds):
        nt.assert_array_equal(ds["variant_allele"], [["T", "A"] for _ in range(7)])

    def test_sample_id(self, ds):
        nt.assert_array_equal(ds["sample_id"], ["s0", "s1", "s2"])


class TestExample:
    """
    .bim file looks like this:

    1       1_10    0       10      A       GG
    1       1_20    0       20      TTT       C

    Definition: https://www.cog-genomics.org/plink/1.9/formats#bim
    Chromosome code (either an integer, or 'X'/'Y'/'XY'/'MT'; '0'
        indicates unknown) or name
    Variant identifier
    Position in morgans or centimorgans (safe to use dummy value of '0')
    Base-pair coordinate (1-based; limited to 231-2)
    Allele 1 (corresponding to clear bits in .bed; usually minor)
    Allele 2 (corresponding to set bits in .bed; usually major)


    Note, the VCF interpretation here can be derived by running
    plink1.9 --bfile tests/data/plink/example --export vcf
    """

    @pytest.fixture(scope="class")
    def ds(self, tmp_path_factory):
        path = "tests/data/plink/example"
        out = tmp_path_factory.mktemp("data") / "example.plink.zarr"
        plink.convert(path, out)
        return sg.load_dataset(out)

    def test_sample_ids(self, ds):
        nt.assert_array_equal(ds.sample_id, [f"ind{j}" for j in range(10)])

    def test_variant_position(self, ds):
        nt.assert_array_equal(ds.variant_position, [10, 20])

    def test_variant_allele(self, ds):
        nt.assert_array_equal(ds.variant_allele, [["GG", "A"], ["C", "TTT"]])

    def test_variant_length(self, ds):
        nt.assert_array_equal(ds.variant_length, [2, 1])

    def test_contig_id(self, ds):
        """Test that contig identifiers are correctly extracted and stored."""
        nt.assert_array_equal(ds.contig_id, ["1"])

    def test_variant_contig(self, ds):
        """Test that variant to contig mapping is correctly stored."""
        nt.assert_array_equal(
            ds.variant_contig, [0, 0]
        )  # Both variants on chromosome 1

    def test_genotypes(self, ds):
        call_genotype = ds.call_genotype.values
        assert call_genotype.shape == (2, 10, 2)
        nt.assert_array_equal(
            call_genotype,
            [
                [
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
                [
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                ],
            ],
        )


class TestSimulatedExample:
    @pytest.fixture(scope="class")
    def ds(self, tmp_path_factory):
        path = "tests/data/plink/plink_sim_10s_100v_10pmiss"
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
                [0, 1],
                [0, 1],
                [1, 1],
                [1, 1],
                [-1, -1],
                [0, 0],
                [0, 0],
                [1, 1],
                [0, 0],
                [0, 0],
            ]
        )
        gt = ds["call_genotype"].values
        actual = gt[tuple(idx.T)]
        # print(actual)
        # print(expected)
        # FIXME not working
        nt.assert_array_equal(actual, expected)

    def test_contig_arrays(self, ds):
        """Test that contig arrays are correctly created and filled."""
        # Verify contig_id exists and is a string array
        assert hasattr(ds, "contig_id")
        assert ds.contig_id.dtype == np.dtype("O")

        # Verify variant_contig exists and contains integer indices
        assert hasattr(ds, "variant_contig")
        assert ds.variant_contig.dtype == np.dtype("int8")
        assert ds.variant_contig.shape[0] == 100  # 100 variants

        # Verify mapping between variant_contig and contig_id
        # For each unique contig index, check at least one corresponding variant exists
        for i, contig in enumerate(ds.contig_id.values):
            assert np.any(
                ds.variant_contig.values == i
            ), f"No variants found for contig {contig}"

    # @pytest.mark.xfail
    @pytest.mark.parametrize(
        ("variants_chunk_size", "samples_chunk_size"),
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
        path = "tests/data/plink/plink_sim_10s_100v_10pmiss"
        out = tmp_path / "example.zarr"
        plink.convert(
            path,
            out,
            variants_chunk_size=variants_chunk_size,
            samples_chunk_size=samples_chunk_size,
            worker_processes=worker_processes,
        )
        ds2 = sg.load_dataset(out)
        # Drop the region_index as it is chunk dependent
        ds = ds.drop_vars("region_index")
        ds2 = ds2.drop_vars("region_index")
        xt.assert_equal(ds, ds2)
        root = zarr.open(out, mode="r")
        v = variants_chunk_size
        s = samples_chunk_size
        assert root["call_genotype"].chunks == (v, s, 2)
        assert root["call_genotype_phased"].chunks == (v, s)
        assert root["variant_position"].chunks == (v,)
        assert root["variant_contig"].chunks == (v,)


def validate(bed_path, zarr_path):
    root = zarr.open(store=zarr_path, mode="r")
    call_genotype = root["call_genotype"][:]

    bed = bed_reader.open_bed(bed_path + ".bed", count_A1=True, num_threads=1)

    assert call_genotype.shape[0] == bed.sid_count
    assert call_genotype.shape[1] == bed.iid_count
    bed_genotypes = bed.read(dtype="int8").T
    assert call_genotype.shape[0] == bed_genotypes.shape[0]
    assert call_genotype.shape[1] == bed_genotypes.shape[1]
    assert call_genotype.shape[2] == 2

    row_id = 0
    for bed_row, zarr_row in zip(bed_genotypes, call_genotype):
        # print("ROW", row_id)
        # print(bed_row, zarr_row)
        row_id += 1
        for bed_call, zarr_call in zip(bed_row, zarr_row):
            if bed_call == -127:
                assert list(zarr_call) == [-1, -1]
            elif bed_call == 0:
                assert list(zarr_call) == [0, 0]
            elif bed_call == 1:
                assert list(zarr_call) == [0, 1]
            elif bed_call == 2:
                assert list(zarr_call) == [1, 1]
            else:  # pragma no cover
                raise AssertionError(f"Unexpected bed call {bed_call}")


@pytest.mark.parametrize(
    ("variants_chunk_size", "samples_chunk_size"),
    [
        (10, 1),
        (10, 10),
        (33, 3),
        (99, 10),
        (3, 10),
    ],
)
@pytest.mark.parametrize("worker_processes", [0])
def test_by_validating(
    tmp_path, variants_chunk_size, samples_chunk_size, worker_processes
):
    path = "tests/data/plink/plink_sim_10s_100v_10pmiss"
    out = tmp_path / "example.zarr"
    plink.convert(
        path,
        out,
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
        worker_processes=worker_processes,
    )
    validate(path, out)


class TestMultipleContigs:
    """Test handling of multiple contigs in PLINK files."""

    @pytest.fixture(scope="class")
    def multi_contig_bed_path(self, tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("data")
        path = tmp_path / "multi_contig.bed"
        # 6 sites x 4 samples
        # - 2 variants on chromosome 1
        # - 1 variant on chromosome 2
        # - 2 variants on chromosome X
        # - 1 variant on chromosome Y
        dosages = np.array(
            [
                [0, 1, 2, 0],  # chr1
                [1, 0, 2, 1],  # chr1
                [0, 0, 0, 2],  # chr2
                [2, 0, 1, 0],  # chrX
                [1, 1, 1, 1],  # chrX
                [0, 2, 0, 2],  # chrY
            ],
            dtype=np.int8,
        )
        bed_reader.to_bed(path, dosages.T)

        bim_path = path.with_suffix(".bim")
        with open(bim_path, "w") as f:
            # Format: chr, variant_id, genetic_dist, pos, allele1, allele2
            f.write("1\t1_10\t0\t10\tA\tG\n")
            f.write("1\t1_20\t0\t20\tT\tC\n")
            f.write("2\t2_10\t0\t10\tA\tG\n")
            f.write("X\tX_10\t0\t10\tC\tT\n")
            f.write("X\tX_20\t0\t20\tG\tA\n")
            f.write("Y\tY_10\t0\t10\tT\tC\n")

        fam_path = path.with_suffix(".fam")
        with open(fam_path, "w") as f:
            for i in range(4):
                # Format: fam_id, ind_id, pat_id, mat_id, sex, phenotype
                f.write(f"fam{i} ind{i} 0 0 0 -9\n")

        return path.with_suffix("")

    @pytest.fixture(scope="class")
    def ds(self, tmp_path_factory, multi_contig_bed_path):
        tmp_path = tmp_path_factory.mktemp("data")
        zarr_path = tmp_path / "multi_contig.plink.zarr"
        plink.convert(multi_contig_bed_path, zarr_path)
        return sg.load_dataset(zarr_path)

    def test_contig_ids(self, ds):
        nt.assert_array_equal(ds.contig_id, ["1", "2", "X", "Y"])

    def test_variant_allele(self, ds):
        nt.assert_array_equal(
            ds.variant_allele,
            [
                ["G", "A"],
                ["C", "T"],
                ["G", "A"],
                ["T", "C"],
                ["A", "G"],
                ["C", "T"],
            ],
        )

    def test_variant_contig(self, ds):
        nt.assert_array_equal(
            ds.variant_contig, [0, 0, 1, 2, 2, 3]
        )  # Each variant mapped to its correct contig index

    def test_genotypes(self, ds):
        call_genotype = ds.call_genotype.values
        assert call_genotype.shape == (6, 4, 2)
        nt.assert_array_equal(
            call_genotype,
            [
                [[0, 0], [0, 1], [1, 1], [0, 0]],  # chr1
                [[0, 1], [0, 0], [1, 1], [0, 1]],  # chr1
                [[0, 0], [0, 0], [0, 0], [1, 1]],  # chr2
                [[1, 1], [0, 0], [0, 1], [0, 0]],  # chrX
                [[0, 1], [0, 1], [0, 1], [0, 1]],  # chrX
                [[0, 0], [1, 1], [0, 0], [1, 1]],  # chrY
            ],
        )

    def test_variant_position(self, ds):
        nt.assert_array_equal(ds.variant_position, [10, 20, 10, 10, 20, 10])

    def test_variant_length(self, ds):
        nt.assert_array_equal(
            ds.variant_length,
            [1, 1, 1, 1, 1, 1],
        )


@pytest.mark.parametrize(
    "prefix",
    [
        "tests/data/plink/example",
        "tests/data/plink/example_with_fam",
        "tests/data/plink/plink_sim_10s_100v_10pmiss",
    ],
)
def test_against_plinks_vcf_output(prefix, tmp_path):
    vcf_path = prefix + ".vcf"
    plink_zarr = tmp_path / "plink.zarr"
    vcf_zarr = tmp_path / "vcf.zarr"
    plink.convert(prefix, plink_zarr)
    vcf.convert([vcf_path], vcf_zarr)
    ds1 = sg.load_dataset(plink_zarr)
    ds2 = (
        sg.load_dataset(vcf_zarr)
        .drop_dims("filters")
        .drop_vars(["variant_quality", "variant_PR", "contig_length"])
    )
    xt.assert_equal(ds1, ds2)
