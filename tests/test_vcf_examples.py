import numpy as np
import numpy.testing as nt
import xarray.testing as xt
import pytest
import sgkit as sg
import zarr

from bio2zarr import vcf


class TestSmallExample:
    data_path = "tests/data/vcf/sample.vcf.gz"

    @pytest.fixture(scope="class")
    def ds(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("data") / "example.vcf.zarr"
        vcf.convert_vcf([self.data_path], out)
        return sg.load_dataset(out)

    def test_filters(self, ds):
        nt.assert_array_equal(ds["filter_id"], ["PASS", "s50", "q10"])
        nt.assert_array_equal(
            ds["variant_filter"],
            [
                [False, False, False],
                [False, False, False],
                [True, False, False],
                [False, False, True],
                [True, False, False],
                [True, False, False],
                [True, False, False],
                [False, False, False],
                [True, False, False],
            ],
        )

    def test_contigs(self, ds):
        nt.assert_array_equal(ds["contig_id"], ["19", "20", "X"])
        assert "contig_length" not in ds
        nt.assert_array_equal(ds["variant_contig"], [0, 0, 1, 1, 1, 1, 1, 1, 2])

    def test_position(self, ds):
        nt.assert_array_equal(
            ds["variant_position"],
            [111, 112, 14370, 17330, 1110696, 1230237, 1234567, 1235237, 10],
        )

    def test_int_info_fields(self, ds):
        nt.assert_array_equal(
            ds["variant_NS"],
            [-1, -1, 3, 3, 2, 3, 3, -1, -1],
        )
        nt.assert_array_equal(
            ds["variant_AN"],
            [-1, -1, -1, -1, -1, -1, 6, -1, -1],
        )

        nt.assert_array_equal(
            ds["variant_AC"],
            [
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [3, 1],
                [-1, -1],
                [-1, -1],
            ],
        )

    def test_float_info_fields(self, ds):
        missing = vcf.FLOAT32_MISSING
        fill = vcf.FLOAT32_FILL
        variant_AF = np.array(
            [
                [missing, missing],
                [missing, missing],
                [0.5, fill],
                [0.017, fill],
                [0.333, 0.667],
                [missing, missing],
                [missing, missing],
                [missing, missing],
                [missing, missing],
            ],
            dtype=np.float32,
        )
        values = ds["variant_AF"].values
        nt.assert_array_almost_equal(values, variant_AF, 3)
        nans = np.isnan(variant_AF)
        nt.assert_array_equal(
            variant_AF.view(np.int32)[nans], values.view(np.int32)[nans]
        )

    def test_string_info_fields(self, ds):
        nt.assert_array_equal(
            ds["variant_AA"],
            [
                ".",
                ".",
                ".",
                ".",
                "T",
                "T",
                "G",
                ".",
                ".",
            ],
        )

    def test_flag_info_fields(self, ds):
        nt.assert_array_equal(
            ds["variant_DB"],
            [
                False,
                False,
                True,
                False,
                True,
                False,
                False,
                False,
                False,
            ],
        )

    def test_allele(self, ds):
        fill = vcf.STR_FILL
        nt.assert_array_equal(
            ds["variant_allele"].values.tolist(),
            [
                ["A", "C", fill, fill],
                ["A", "G", fill, fill],
                ["G", "A", fill, fill],
                ["T", "A", fill, fill],
                ["A", "G", "T", fill],
                ["T", fill, fill, fill],
                ["G", "GA", "GAC", fill],
                ["T", fill, fill, fill],
                ["AC", "A", "ATG", "C"],
            ],
        )
        assert ds["variant_allele"].dtype == "O"

    def test_id(self, ds):
        nt.assert_array_equal(
            ds["variant_id"].values.tolist(),
            [".", ".", "rs6054257", ".", "rs6040355", ".", "microsat1", ".", "rsTest"],
        )
        assert ds["variant_id"].dtype == "O"
        nt.assert_array_equal(
            ds["variant_id_mask"],
            [True, True, False, True, False, True, False, True, False],
        )

    def test_samples(self, ds):
        nt.assert_array_equal(ds["sample_id"], ["NA00001", "NA00002", "NA00003"])

    def test_call_genotype(self, ds):
        call_genotype = np.array(
            [
                [[0, 0], [0, 0], [0, 1]],
                [[0, 0], [0, 0], [0, 1]],
                [[0, 0], [1, 0], [1, 1]],
                [[0, 0], [0, 1], [0, 0]],
                [[1, 2], [2, 1], [2, 2]],
                [[0, 0], [0, 0], [0, 0]],
                [[0, 1], [0, 2], [-1, -1]],
                [[0, 0], [0, 0], [-1, -1]],
                # FIXME this depends on "mixed ploidy" interpretation.
                [[0, -2], [0, 1], [0, 2]],
            ],
            dtype="i1",
        )
        nt.assert_array_equal(ds["call_genotype"], call_genotype)
        nt.assert_array_equal(ds["call_genotype_mask"], call_genotype < 0)

    def test_call_genotype_phased(self, ds):
        call_genotype_phased = np.array(
            [
                [True, True, False],
                [True, True, False],
                [True, True, False],
                [True, True, False],
                [True, True, False],
                [True, True, False],
                [False, False, False],
                [False, True, False],
                [True, False, True],
            ],
            dtype=bool,
        )
        nt.assert_array_equal(ds["call_genotype_phased"], call_genotype_phased)

    def test_call_DP(self, ds):
        call_DP = [
            [-1, -1, -1],
            [-1, -1, -1],
            [1, 8, 5],
            [3, 5, 3],
            [6, 0, 4],
            [-1, 4, 2],
            [4, 2, 3],
            [-1, -1, -1],
            [-1, -1, -1],
        ]
        nt.assert_array_equal(ds["call_DP"], call_DP)

    def test_call_HQ(self, ds):
        call_HQ = [
            [[10, 15], [10, 10], [3, 3]],
            [[10, 10], [10, 10], [3, 3]],
            [[51, 51], [51, 51], [-1, -1]],
            [[58, 50], [65, 3], [-1, -1]],
            [[23, 27], [18, 2], [-1, -1]],
            [[56, 60], [51, 51], [-1, -1]],
            [[-1, -1], [-1, -1], [-1, -1]],
            [[-1, -1], [-1, -1], [-1, -1]],
            [[-1, -1], [-1, -1], [-1, -1]],
        ]
        nt.assert_array_equal(ds["call_HQ"], call_HQ)

    def test_no_genotypes(self, ds, tmp_path):
        path = "tests/data/vcf/sample_no_genotypes.vcf.gz"
        out = tmp_path / "example.vcf.zarr"
        vcf.convert_vcf([path], out)
        ds2 = sg.load_dataset(out)
        assert len(ds2["sample_id"]) == 0
        for col in ds:
            if col != "sample_id" and not col.startswith("call_"):
                xt.assert_equal(ds[col], ds2[col])

    @pytest.mark.parametrize(
        ["chunk_length", "chunk_width", "y_chunks", "x_chunks"],
        [
            (1, 1, (1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1)),
            (2, 2, (2, 2, 2, 2, 1), (2, 1)),
            (3, 3, (3, 3, 3), (3,)),
            (4, 3, (4, 4, 1), (3,)),
        ],
    )
    def test_chunk_size(
        self, ds, tmp_path, chunk_length, chunk_width, y_chunks, x_chunks
    ):
        out = tmp_path / "example.vcf.zarr"
        vcf.convert_vcf(
            [self.data_path], out, chunk_length=chunk_length, chunk_width=chunk_width
        )
        ds2 = sg.load_dataset(out)
        # print(ds2.call_genotype.values)
        # print(ds.call_genotype.values)
        xt.assert_equal(ds, ds2)
        assert ds2.call_DP.chunks == (y_chunks, x_chunks)
        assert ds2.call_GQ.chunks == (y_chunks, x_chunks)
        assert ds2.call_HQ.chunks == (y_chunks, x_chunks, (2,))
        assert ds2.call_genotype.chunks == (y_chunks, x_chunks, (2,))
        assert ds2.call_genotype_mask.chunks == (y_chunks, x_chunks, (2,))
        assert ds2.call_genotype_phased.chunks == (y_chunks, x_chunks)
        assert ds2.variant_AA.chunks == (y_chunks,)
        assert ds2.variant_AC.chunks == (y_chunks, (2,))
        assert ds2.variant_AF.chunks == (y_chunks, (2,))
        assert ds2.variant_DB.chunks == (y_chunks,)
        assert ds2.variant_DP.chunks == (y_chunks,)
        assert ds2.variant_NS.chunks == (y_chunks,)
        assert ds2.variant_allele.chunks == (y_chunks, (4,))
        assert ds2.variant_contig.chunks == (y_chunks,)
        assert ds2.variant_filter.chunks == (y_chunks, (3,))
        assert ds2.variant_id.chunks == (y_chunks,)
        assert ds2.variant_id_mask.chunks == (y_chunks,)
        assert ds2.variant_position.chunks == (y_chunks,)
        assert ds2.variant_quality.chunks == (y_chunks,)
        assert ds2.contig_id.chunks == ((3,),)
        assert ds2.filter_id.chunks == ((3,),)
        assert ds2.sample_id.chunks == (x_chunks,)

    @pytest.mark.parametrize("worker_processes", [0, 1, 2])
    def test_worker_processes(self, ds, tmp_path, worker_processes):
        out = tmp_path / "example.vcf.zarr"
        vcf.convert_vcf(
            [self.data_path], out, chunk_length=3, worker_processes=worker_processes
        )
        ds2 = sg.load_dataset(out)
        xt.assert_equal(ds, ds2)

    @pytest.mark.parametrize("worker_processes", [0, 1, 2])
    def test_full_pipeline(self, ds, tmp_path, worker_processes):
        exploded = tmp_path / "example.exploded"
        vcf.explode(
            [self.data_path], exploded, worker_processes=worker_processes,
        )
        schema = tmp_path / "schema.json"
        with open(schema, "w") as f:
            vcf.generate_spec(exploded, f)
        out = tmp_path / "example.zarr"
        vcf.to_zarr(exploded, out, schema, worker_processes=worker_processes)
        ds2 = sg.load_dataset(out)
        xt.assert_equal(ds, ds2)


@pytest.mark.parametrize(
    "name",
    [
        "sample.vcf.gz",
        "sample_no_genotypes.vcf.gz",
        "info_field_type_combos.vcf.gz",
    ],
)
def test_by_validating(name, tmp_path):
    path = f"tests/data/vcf/{name}"
    out = tmp_path / "test.zarr"
    vcf.convert_vcf([path], out, worker_processes=0)
    vcf.validate(path, out)
