import collections
import pathlib
import re
from unittest import mock

import numpy as np
import numpy.testing as nt
import pytest
import sgkit as sg
import xarray.testing as xt

from bio2zarr import constants, provenance, vcz_verification
from bio2zarr import vcf as vcf_mod


def assert_dataset_equal(ds1, ds2, drop_vars=None):
    if drop_vars is None:
        xt.assert_equal(ds1, ds2)
    else:
        xt.assert_equal(ds1.drop_vars(drop_vars), ds2.drop_vars(drop_vars))


class TestSmallExample:
    data_path = "tests/data/vcf/sample.vcf.gz"

    @pytest.fixture(scope="class")
    def ds(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("data") / "example.vcf.zarr"
        vcf_mod.convert([self.data_path], out)
        return sg.load_dataset(out)

    def test_filters(self, ds):
        nt.assert_array_equal(ds["filter_id"], ["PASS", "s50", "q10"])
        nt.assert_array_equal(
            ds["filter_description"],
            [
                "All filters passed",
                "Less than 50% of samples have data",
                "Quality below 10",
            ],
        )
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

    def test_vcf_meta_information(self, ds):
        assert ds.attrs["vcf_meta_information"] == [
            ["fileformat", "VCFv4.0"],
            ["fileDate", "20090805"],
            ["source", "myImputationProgramV3.1"],
            ["reference", "1000GenomesPilot-NCBI36"],
            ["phasing", "partial"],
            ["ALT", '<ID=DEL:ME:ALU,Description="Deletion of ALU element">'],
            ["ALT", '<ID=CNV,Description="Copy number variable region">'],
            ["bcftools_viewVersion", "1.11+htslib-1.11-4"],
            [
                "bcftools_viewCommand",
                "view -O b sample.vcf.gz; Date=Tue Feb 27 14:41:07 2024",
            ],
            ["bcftools_viewCommand", "view sample.bcf; Date=Wed Mar 27 11:42:16 2024"],
        ]

    def test_source(self, ds):
        assert ds.attrs["source"] == f"bio2zarr-{provenance.__version__}"

    def test_contigs(self, ds):
        nt.assert_array_equal(ds["contig_id"], ["19", "20", "X"])
        assert "contig_length" not in ds
        nt.assert_array_equal(ds["variant_contig"], [0, 0, 1, 1, 1, 1, 1, 1, 2])

    def test_position(self, ds):
        nt.assert_array_equal(
            ds["variant_position"],
            [111, 112, 14370, 17330, 1110696, 1230237, 1234567, 1235237, 10],
        )

    def test_length(self, ds):
        nt.assert_array_equal(ds["variant_length"], [1, 1, 1, 1, 1, 1, 1, 1, 2])

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
        missing = constants.FLOAT32_MISSING
        fill = constants.FLOAT32_FILL
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
        fill = constants.STR_FILL
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
        vcf_mod.convert([path], out)
        ds2 = sg.load_dataset(out)
        assert len(ds2["sample_id"]) == 0
        for field_name in ds:
            if field_name != "sample_id" and not field_name.startswith("call_"):
                xt.assert_equal(ds[field_name], ds2[field_name])

    @pytest.mark.parametrize(
        ("variants_chunk_size", "samples_chunk_size", "y_chunks", "x_chunks"),
        [
            (1, 1, (1, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1)),
            (2, 2, (2, 2, 2, 2, 1), (2, 1)),
            (3, 3, (3, 3, 3), (3,)),
            (4, 3, (4, 4, 1), (3,)),
        ],
    )
    def test_chunk_size(
        self, ds, tmp_path, variants_chunk_size, samples_chunk_size, y_chunks, x_chunks
    ):
        out = tmp_path / "example.vcf.zarr"
        vcf_mod.convert(
            [self.data_path],
            out,
            variants_chunk_size=variants_chunk_size,
            samples_chunk_size=samples_chunk_size,
        )
        ds2 = sg.load_dataset(out)
        # print(ds2.call_genotype.values)
        # print(ds.call_genotype.values)
        assert_dataset_equal(ds, ds2, drop_vars=["region_index"])
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
    @pytest.mark.parametrize("rotate", [0, 1, 2])
    def test_split(self, ds, tmp_path, worker_processes, rotate):
        out = tmp_path / "example.vcf.zarr"
        split_path = pathlib.Path(self.data_path + ".3.split")
        files = collections.deque(sorted(list(split_path.glob("*.vcf.gz"))))
        # Rotate the list to check we are OK with different orderings
        files.rotate(rotate)
        assert len(files) == 3
        vcf_mod.convert(files, out, worker_processes=worker_processes)
        ds2 = sg.load_dataset(out)
        xt.assert_equal(ds, ds2)

    @pytest.mark.parametrize("worker_processes", [0, 1, 2])
    def test_full_pipeline(self, ds, tmp_path, worker_processes):
        exploded = tmp_path / "example.exploded"
        vcf_mod.explode(
            exploded,
            [self.data_path],
            worker_processes=worker_processes,
        )
        schema = tmp_path / "schema.json"
        with open(schema, "w") as f:
            vcf_mod.mkschema(exploded, f)
        out = tmp_path / "example.zarr"
        vcf_mod.encode(exploded, out, schema, worker_processes=worker_processes)
        ds2 = sg.load_dataset(out)
        xt.assert_equal(ds, ds2)

    @pytest.mark.parametrize("max_variant_chunks", [1, 2, 3])
    @pytest.mark.parametrize("variants_chunk_size", [1, 2, 3])
    def test_max_variant_chunks(
        self, ds, tmp_path, max_variant_chunks, variants_chunk_size
    ):
        exploded = tmp_path / "example.exploded"
        vcf_mod.explode(exploded, [self.data_path])
        out = tmp_path / "example.zarr"
        vcf_mod.encode(
            exploded,
            out,
            variants_chunk_size=variants_chunk_size,
            max_variant_chunks=max_variant_chunks,
        )
        ds2 = sg.load_dataset(out)
        assert_dataset_equal(
            ds.isel(variants=slice(None, variants_chunk_size * max_variant_chunks)),
            ds2,
            drop_vars=["region_index"],
        )

    @pytest.mark.parametrize("worker_processes", [0, 1, 2])
    def test_worker_processes(self, ds, tmp_path, worker_processes):
        out = tmp_path / "example.vcf.zarr"
        vcf_mod.convert(
            [self.data_path],
            out,
            variants_chunk_size=3,
            worker_processes=worker_processes,
        )
        ds2 = sg.load_dataset(out)
        assert_dataset_equal(ds, ds2, drop_vars=["region_index"])

    def test_inspect(self, tmp_path):
        # TODO pretty weak test, we should be doing this better somewhere else
        out = tmp_path / "example.vcf.zarr"
        vcf_mod.convert(
            [self.data_path],
            out,
            variants_chunk_size=3,
        )
        data = vcf_mod.inspect(out)
        assert len(data) > 0
        for row in data:
            assert "name" in row

    @pytest.mark.parametrize(
        "path",
        [
            "tests/data/vcf/sample_missing_contig.vcf.gz",
            "tests/data/vcf/sample_missing_contig.bcf",
            "tests/data/vcf/sample_missing_contig_csi.vcf.gz",
        ],
    )
    def test_missing_contig_vcf(self, ds, tmp_path, path):
        # 20 has been removed from the header. The datasets is the same,
        # but the ordering of contigs has been permuted. This seems to be the
        # sample across VCF and BCF with tabix and VSI indexes
        zarr_path = tmp_path / "zarr"
        vcf_mod.convert([path], zarr_path)
        ds2 = sg.load_dataset(zarr_path)
        contig_id_2 = ["19", "X", "20"]
        assert list(ds2["contig_id"].values) == contig_id_2
        for id1, contig in enumerate(["19", "20", "X"]):
            ds_c1 = ds.isel(variants=ds["variant_contig"].values == id1)
            id2 = contig_id_2.index(contig)
            ds_c2 = ds2.isel(variants=ds2["variant_contig"].values == id2)
            drop_vars = ["contig_id", "variant_contig", "region_index"]
            assert_dataset_equal(ds_c1, ds_c2, drop_vars=drop_vars)

    def test_vcf_dimensions(self, ds):
        assert ds.call_genotype.dims == ("variants", "samples", "ploidy")
        assert ds.call_genotype_mask.dims == ("variants", "samples", "ploidy")
        assert ds.call_genotype_phased.dims == ("variants", "samples")
        assert ds.call_HQ.dims == ("variants", "samples", "FORMAT_HQ_dim")
        assert ds.call_DP.dims == ("variants", "samples")
        assert ds.call_GQ.dims == ("variants", "samples")
        assert ds.variant_AA.dims == ("variants",)
        assert ds.variant_NS.dims == ("variants",)
        assert ds.variant_AN.dims == ("variants",)
        assert ds.variant_AC.dims == ("variants", "INFO_AC_dim")
        assert ds.variant_AF.dims == ("variants", "INFO_AF_dim")
        assert ds.variant_DP.dims == ("variants",)
        assert ds.variant_DB.dims == ("variants",)
        assert ds.variant_H2.dims == ("variants",)
        assert ds.variant_position.dims == ("variants",)

    @pytest.mark.parametrize(
        ("field", "description"),
        [
            ("variant_NS", "Number of Samples With Data"),
            ("variant_AN", "Total number of alleles in called genotypes"),
            (
                "variant_AC",
                "Allele count in genotypes, for each ALT allele, "
                "in the same order as listed",
            ),
            ("variant_DP", "Total Depth"),
            ("variant_AF", "Allele Frequency"),
            ("variant_AA", "Ancestral Allele"),
            ("variant_DB", "dbSNP membership, build 129"),
            ("variant_H2", "HapMap2 membership"),
            ("call_GQ", "Genotype Quality"),
            ("call_DP", "Read Depth"),
            ("call_HQ", "Haplotype Quality"),
        ],
    )
    def test_vcf_field_description(self, ds, field, description):
        assert ds[field].attrs["description"] == description

    def test_region_index(self, ds):
        assert ds["region_index"].chunks == ((3,), (6,))
        region_index = np.array(
            [
                [0, 0, 111, 112, 112, 2],
                [0, 1, 14370, 1235237, 1235237, 6],
                [0, 2, 10, 10, 11, 1],
            ]
        )
        nt.assert_array_equal(ds["region_index"], region_index)

    def test_small_example_all_missing_gts(self, ds, tmp_path_factory):
        data_path = "tests/data/vcf/sample_all_missing_gts.vcf.gz"
        out = tmp_path_factory.mktemp("data") / "example.vcf.zarr"
        vcf_mod.convert([data_path], out, worker_processes=0)
        ds2 = sg.load_dataset(out)

        assert_dataset_equal(
            ds,
            ds2,
            drop_vars=["call_genotype", "call_genotype_mask", "call_genotype_phased"],
        )
        gt1 = ds["call_genotype"].values
        gt1[1] = -1
        nt.assert_array_equal(gt1, ds2["call_genotype"].values)
        m1 = ds["call_genotype_mask"].values
        m1[1] = True
        nt.assert_array_equal(m1, ds2["call_genotype_mask"].values)
        p1 = ds["call_genotype_phased"].values
        # NOTE: Not sure this is the correct behaviour, but testing here anyway
        # to keep a record that this is what we're doing
        p1[1] = True
        nt.assert_array_equal(p1, ds2["call_genotype_phased"].values)

    def test_missing_dependency(self, tmp_path):
        with mock.patch(
            "importlib.import_module",
            side_effect=ImportError("No module named 'cyvcf2'"),
        ):
            with pytest.raises(ImportError) as exc_info:
                vcf_mod.convert(
                    ["tests/data/vcf/sample.vcf.gz"],
                    tmp_path / "example.vcf.zarr",
                    worker_processes=0,  # Synchronous mode so the mock works
                )
            assert (
                "This process requires the optional cyvcf2 module. Install "
                "it with: pip install bio2zarr[vcf]" in str(exc_info.value)
            )


class TestSmallExampleLocalAlleles:
    data_path = "tests/data/vcf/sample.vcf.gz"

    @pytest.fixture(scope="class")
    def ds(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("data") / "example.vcf.zarr"
        vcf_mod.convert([self.data_path], out, local_alleles=True)
        return sg.load_dataset(out)

    def test_call_LA(self, ds):
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

        call_LA = np.array(
            [
                [[0, -2], [0, -2], [0, 1]],
                [[0, -2], [0, -2], [0, 1]],
                [[0, -2], [0, 1], [1, -2]],
                [[0, -2], [0, 1], [0, -2]],
                [[1, 2], [1, 2], [2, -2]],
                [[0, -2], [0, -2], [0, -2]],
                [[0, 1], [0, 2], [-2, -2]],
                [[0, -2], [0, -2], [-2, -2]],
                [[0, -2], [0, 1], [0, 2]],
            ],
        )
        nt.assert_array_equal(ds.call_LA.values, call_LA)

    @pytest.mark.parametrize("field", ["call_LPL", "call_LAD"])
    def test_no_localised_fields(self, ds, field):
        assert field not in ds


class TestTriploidExample:
    @pytest.fixture(scope="class", params=["triploid", "triploid2", "triploid3"])
    def ds(self, tmp_path_factory, request):
        data_path = f"tests/data/vcf/{request.param}.vcf.gz"
        out = tmp_path_factory.mktemp("data") / "example.vcf.zarr"
        vcf_mod.convert([data_path], out, local_alleles=False)
        return sg.load_dataset(out)

    @pytest.mark.parametrize("name", ["triploid", "triploid2", "triploid3"])
    def test_error_with_local_alleles(self, tmp_path_factory, name):
        data_path = f"tests/data/vcf/{name}.vcf.gz"
        out = tmp_path_factory.mktemp("data") / "example.vcf.zarr"
        with pytest.raises(
            ValueError, match=re.escape("Local alleles only supported on diploid")
        ):
            vcf_mod.convert([data_path], out, local_alleles=True)

    def test_ok_without_local_alleles(self, ds):
        nt.assert_array_equal(ds.call_genotype.values, [[[0, 0, 0]]])


class TestWithGtHeaderNoGenotypes:
    data_path = "tests/data/vcf/sample_no_genotypes_with_gt_header.vcf.gz"

    @pytest.fixture(scope="class")
    def ds(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("data") / "example.vcf.zarr"
        vcf_mod.convert([self.data_path], out, worker_processes=0)
        return sg.load_dataset(out)

    def test_gts(self, ds):
        assert "call_genotype" not in ds


class TestChr22Example:
    data_path = "tests/data/vcf/chr22.vcf.gz"

    @pytest.fixture(scope="class")
    def ds(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("data") / "example.vcf.zarr"
        vcf_mod.convert([self.data_path], out, worker_processes=0)
        return sg.load_dataset(out)

    def test_call_SB(self, ds):
        # fixes https://github.com/sgkit-dev/bio2zarr/issues/355
        assert ds.call_SB.dims == ("variants", "samples", "FORMAT_SB_dim")
        assert ds.call_SB.shape == (100, 100, 4)


class Test1000G2020Example:
    data_path = "tests/data/vcf/1kg_2020_chrM.vcf.gz"

    @pytest.fixture(scope="class")
    def ds(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("data") / "example.vcf.zarr"
        vcf_mod.convert([self.data_path], out, worker_processes=0)
        return sg.load_dataset(out)

    def test_position(self, ds):
        # fmt: off
        pos = [
            26, 35, 40, 41, 42, 46, 47, 51, 52, 53, 54, 55, 56,
            57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
        ]
        # fmt: on
        nt.assert_array_equal(ds.variant_position.values, pos)

    def test_alleles(self, ds):
        alleles = [
            ["C", "T", "", "", ""],
            ["G", "A", "", "", ""],
            ["T", "C", "", "", ""],
            ["C", "T", "CT", "", ""],
            ["T", "TC", "C", "TG", ""],
            ["T", "C", "", "", ""],
            ["G", "A", "", "", ""],
            ["T", "C", "", "", ""],
            ["T", "C", "", "", ""],
            ["G", "A", "", "", ""],
            ["G", "A", "", "", ""],
            ["TA", "TAA", "T", "CA", "AA"],
            ["ATT", "*", "ATTT", "ACTT", "A"],
            ["T", "C", "G", "*", "TC"],
            ["T", "A", "C", "*", ""],
            ["T", "A", "", "", ""],
            ["T", "A", "", "", ""],
            ["C", "A", "T", "", ""],
            ["G", "A", "", "", ""],
            ["T", "C", "A", "", ""],
            ["C", "T", "CT", "A", ""],
            ["TG", "T", "CG", "TGG", "TCGG"],
            ["G", "T", "*", "A", ""],
        ]
        nt.assert_array_equal(ds.variant_allele.values, alleles)

    def test_variant_MLEAC(self, ds):
        MLEAC = np.array(
            [
                [2, -2, -2, -2],
                [2, -2, -2, -2],
                [2, -2, -2, -2],
                [16, 2, -2, -2],
                [10, 4, 2, -2],
                [2, -2, -2, -2],
                [4, -2, -2, -2],
                [4, -2, -2, -2],
                [2, -2, -2, -2],
                [2, -2, -2, -2],
                [2, -2, -2, -2],
                [2, 26, 12, 4],
                [26, 2, 8, 4],
                [11, 6, 4, 2],
                [26, 1, 4, -2],
                [2, -2, -2, -2],
                [2, -2, -2, -2],
                [2, 12, -2, -2],
                [14, -2, -2, -2],
                [4, 2, -2, -2],
                [320, 20, 2, -2],
                [12, 2, 6, 4],
                [18, 12, 4, -2],
            ],
            dtype=np.int16,
        )
        nt.assert_array_equal(ds.variant_MLEAC.values, MLEAC)

    def test_call_AD(self, ds):
        call_AD = [
            [[446, 0, -2, -2, -2], [393, 0, -2, -2, -2], [486, 0, -2, -2, -2]],
            [[446, 0, -2, -2, -2], [393, 0, -2, -2, -2], [486, 0, -2, -2, -2]],
            [[446, 0, -2, -2, -2], [393, 0, -2, -2, -2], [486, 0, -2, -2, -2]],
            [[446, 0, 0, -2, -2], [393, 0, 0, -2, -2], [486, 0, 0, -2, -2]],
            [[446, 0, 0, 0, -2], [393, 0, 0, 0, -2], [486, 0, 0, 0, -2]],
            [[446, 0, -2, -2, -2], [393, 0, -2, -2, -2], [486, 0, -2, -2, -2]],
            [[446, 0, -2, -2, -2], [393, 0, -2, -2, -2], [486, 0, -2, -2, -2]],
            [[446, 0, -2, -2, -2], [393, 0, -2, -2, -2], [486, 0, -2, -2, -2]],
            [[446, 0, -2, -2, -2], [393, 0, -2, -2, -2], [486, 0, -2, -2, -2]],
            [[446, 0, -2, -2, -2], [393, 0, -2, -2, -2], [486, 0, -2, -2, -2]],
            [[446, 0, -2, -2, -2], [393, 0, -2, -2, -2], [486, 0, -2, -2, -2]],
            [[446, 0, 0, 0, 0], [393, 0, 0, 0, 0], [486, 0, 0, 0, 0]],
            [[446, 0, 0, 0, 0], [393, 0, 0, 0, 0], [486, 0, 0, 0, 0]],
            [[446, 0, 0, 0, 0], [393, 0, 0, 0, 0], [486, 0, 0, 0, 0]],
            [[446, 0, 0, 0, -2], [393, 0, 0, 0, -2], [486, 0, 0, 0, -2]],
            [[446, 0, -2, -2, -2], [393, 0, -2, -2, -2], [486, 0, -2, -2, -2]],
            [[446, 0, -2, -2, -2], [393, 0, -2, -2, -2], [486, 0, -2, -2, -2]],
            [[446, 0, 0, -2, -2], [393, 0, 0, -2, -2], [486, 0, 0, -2, -2]],
            [[446, 0, -2, -2, -2], [393, 0, -2, -2, -2], [486, 0, -2, -2, -2]],
            [[446, 0, 0, -2, -2], [393, 0, 0, -2, -2], [486, 0, 0, -2, -2]],
            [[446, 0, 0, 0, -2], [393, 0, 0, 0, -2], [486, 0, 0, 0, -2]],
            [[446, 0, 0, 0, 0], [393, 0, 0, 0, 0], [486, 0, 0, 0, 0]],
            [[446, 0, 0, 0, -2], [393, 0, 0, 0, -2], [486, 0, 0, 0, -2]],
        ]
        nt.assert_array_equal(ds.call_AD.values, call_AD)

    def test_call_PID(self, ds):
        call_PGT = ds["call_PGT"].values
        assert np.all(call_PGT == ".")
        assert call_PGT.shape == (23, 3)


class Test1000G2020ExampleLocalAlleles:
    data_path = "tests/data/vcf/1kg_2020_chrM.vcf.gz"

    @pytest.fixture(scope="class")
    def ds(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("data") / "example.vcf.zarr"
        vcf_mod.convert([self.data_path], out, worker_processes=0, local_alleles=True)
        return sg.load_dataset(out)

    def test_position(self, ds):
        # fmt: off
        pos = [
            26, 35, 40, 41, 42, 46, 47, 51, 52, 53, 54, 55, 56,
            57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
        ]
        # fmt: on
        nt.assert_array_equal(ds.variant_position.values, pos)

    def test_alleles(self, ds):
        alleles = [
            ["C", "T", "", "", ""],
            ["G", "A", "", "", ""],
            ["T", "C", "", "", ""],
            ["C", "T", "CT", "", ""],
            ["T", "TC", "C", "TG", ""],
            ["T", "C", "", "", ""],
            ["G", "A", "", "", ""],
            ["T", "C", "", "", ""],
            ["T", "C", "", "", ""],
            ["G", "A", "", "", ""],
            ["G", "A", "", "", ""],
            ["TA", "TAA", "T", "CA", "AA"],
            ["ATT", "*", "ATTT", "ACTT", "A"],
            ["T", "C", "G", "*", "TC"],
            ["T", "A", "C", "*", ""],
            ["T", "A", "", "", ""],
            ["T", "A", "", "", ""],
            ["C", "A", "T", "", ""],
            ["G", "A", "", "", ""],
            ["T", "C", "A", "", ""],
            ["C", "T", "CT", "A", ""],
            ["TG", "T", "CG", "TGG", "TCGG"],
            ["G", "T", "*", "A", ""],
        ]
        nt.assert_array_equal(ds.variant_allele.values, alleles)

    def test_call_LAD(self, ds):
        call_LAD = [
            [[446, -2], [393, -2], [486, -2]],
            [[446, -2], [393, -2], [486, -2]],
            [[446, -2], [393, -2], [486, -2]],
            [[446, -2], [393, -2], [486, -2]],
            [[446, -2], [393, -2], [486, -2]],
            [[446, -2], [393, -2], [486, -2]],
            [[446, -2], [393, -2], [486, -2]],
            [[446, -2], [393, -2], [486, -2]],
            [[446, -2], [393, -2], [486, -2]],
            [[446, -2], [393, -2], [486, -2]],
            [[446, -2], [393, -2], [486, -2]],
            [[446, -2], [393, -2], [486, -2]],
            [[446, -2], [393, -2], [486, -2]],
            [[446, -2], [393, -2], [486, -2]],
            [[446, -2], [393, -2], [486, -2]],
            [[446, -2], [393, -2], [486, -2]],
            [[446, -2], [393, -2], [486, -2]],
            [[446, -2], [393, -2], [486, -2]],
            [[446, -2], [393, -2], [486, -2]],
            [[446, -2], [393, -2], [486, -2]],
            [[446, -2], [393, -2], [486, -2]],
            [[446, -2], [393, -2], [486, -2]],
            [[446, -2], [393, -2], [486, -2]],
        ]
        nt.assert_array_equal(ds.call_LAD.values, call_LAD)
        assert ds.call_LAD.dims == ("variants", "samples", "local_alleles_AD")

    def test_call_LA(self, ds):
        # All the genotypes are 0/0
        call_LA = np.full((23, 3, 2), -2)
        call_LA[:, :, 0] = 0
        nt.assert_array_equal(ds.call_LA.values, call_LA)
        assert ds.call_LA.dims == ("variants", "samples", "local_alleles")

    def test_call_LPL(self, ds):
        call_LPL = np.tile([0, -2, -2], (23, 3, 1))
        nt.assert_array_equal(ds.call_LPL.values, call_LPL)
        assert ds.call_LPL.dims == ("variants", "samples", "local_genotypes")


class Test1000G2020AnnotationsExample:
    data_path = "tests/data/vcf/1kg_2020_chr20_annotations.bcf"

    @pytest.fixture(scope="class")
    def ds(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("data") / "example.zarr"
        # TODO capture warnings from htslib here
        vcf_mod.convert([self.data_path], out, worker_processes=0)
        return sg.load_dataset(out)

    def test_position(self, ds):
        # fmt: off
        pos = [
            60070, 60083, 60114, 60116, 60137, 60138, 60149, 60181, 60183,
            60254, 60280, 60280, 60286, 60286, 60291, 60291, 60291, 60291,
            60291, 60329, 60331
        ]
        # fmt: on
        nt.assert_array_equal(ds.variant_position.values, pos)

    def test_alleles(self, ds):
        alleles = [
            ["G", "A"],
            ["T", "C"],
            ["T", "C"],
            ["A", "G"],
            ["T", "C"],
            ["T", "A"],
            ["C", "T"],
            ["A", "G"],
            ["A", "G"],
            ["C", "A"],
            ["TTTCCA", "T"],
            ["T", "TTTCCA"],
            ["T", "G"],
            ["TTCCAG", "T"],
            ["G", "T"],
            ["G", "GTCCAT"],
            ["GTCCATTCCAT", "G"],
            ["GTCCAT", "G"],
            ["G", "GTCCATTCCAT"],
            ["C", "G"],
            ["T", "C"],
        ]
        nt.assert_array_equal(ds.variant_allele.values, alleles)

    def test_info_fields(self, ds):
        info_vars = [
            "variant_1000Gp3_AA_GF",
            "variant_1000Gp3_AF",
            "variant_1000Gp3_HomC",
            "variant_1000Gp3_RA_GF",
            "variant_1000Gp3_RR_GF",
            "variant_AC",
            "variant_ACMG_GENE",
            "variant_ACMG_INHRT",
            "variant_ACMG_MIM_GENE",
            "variant_ACMG_PATH",
            "variant_AC_AFR",
            "variant_AC_AFR_unrel",
            "variant_AC_AMR",
            "variant_AC_AMR_unrel",
            "variant_AC_EAS",
            "variant_AC_EAS_unrel",
            "variant_AC_EUR",
            "variant_AC_EUR_unrel",
            "variant_AC_Het",
            "variant_AC_Het_AFR",
            "variant_AC_Het_AFR_unrel",
            "variant_AC_Het_AMR",
            "variant_AC_Het_AMR_unrel",
            "variant_AC_Het_EAS",
            "variant_AC_Het_EAS_unrel",
            "variant_AC_Het_EUR",
            "variant_AC_Het_EUR_unrel",
            "variant_AC_Het_SAS",
            "variant_AC_Het_SAS_unrel",
            "variant_AC_Hom",
            "variant_AC_Hom_AFR",
            "variant_AC_Hom_AFR_unrel",
            "variant_AC_Hom_AMR",
            "variant_AC_Hom_AMR_unrel",
            "variant_AC_Hom_EAS",
            "variant_AC_Hom_EAS_unrel",
            "variant_AC_Hom_EUR",
            "variant_AC_Hom_EUR_unrel",
            "variant_AC_Hom_SAS",
            "variant_AC_Hom_SAS_unrel",
            "variant_AC_SAS",
            "variant_AC_SAS_unrel",
            "variant_AF",
            "variant_AF_AFR",
            "variant_AF_AFR_unrel",
            "variant_AF_AMR",
            "variant_AF_AMR_unrel",
            "variant_AF_EAS",
            "variant_AF_EAS_unrel",
            "variant_AF_EUR",
            "variant_AF_EUR_unrel",
            "variant_AF_SAS",
            "variant_AF_SAS_unrel",
            "variant_AN",
            "variant_ANN",
            "variant_AN_AFR",
            "variant_AN_AFR_unrel",
            "variant_AN_AMR",
            "variant_AN_AMR_unrel",
            "variant_AN_EAS",
            "variant_AN_EAS_unrel",
            "variant_AN_EUR",
            "variant_AN_EUR_unrel",
            "variant_AN_SAS",
            "variant_AN_SAS_unrel",
            "variant_AR_GENE",
            "variant_BaseQRankSum",
            "variant_CADD_phred",
            "variant_CLNDBN",
            "variant_CLNDSDB",
            "variant_CLNDSDBID",
            "variant_CLNSIG",
            "variant_COSMIC_CNT",
            "variant_ClippingRankSum",
            "variant_DP",
            "variant_DS",
            "variant_END",
            "variant_Entrez_gene_id",
            "variant_Essential_gene",
            "variant_ExAC_AF",
            "variant_ExcHet",
            "variant_ExcHet_AFR",
            "variant_ExcHet_AMR",
            "variant_ExcHet_EAS",
            "variant_ExcHet_EUR",
            "variant_ExcHet_SAS",
            "variant_FS",
            "variant_GDI",
            "variant_GDI-Phred",
            "variant_GERP++_NR",
            "variant_GERP++_RS",
            "variant_HWE",
            "variant_HWE_AFR",
            "variant_HWE_AFR_unrel",
            "variant_HWE_AMR",
            "variant_HWE_AMR_unrel",
            "variant_HWE_EAS",
            "variant_HWE_EAS_unrel",
            "variant_HWE_EUR",
            "variant_HWE_EUR_unrel",
            "variant_HWE_SAS",
            "variant_HWE_SAS_unrel",
            "variant_HaplotypeScore",
            "variant_InbreedingCoeff",
            "variant_LOF",
            "variant_LoFtool_score",
            "variant_ME",
            "variant_MGI_mouse_gene",
            "variant_MLEAC",
            "variant_MLEAF",
            "variant_MQ",
            "variant_MQ0",
            "variant_MQRankSum",
            "variant_MutationAssessor_pred",
            "variant_MutationTaster_pred",
            "variant_NEGATIVE_TRAIN_SITE",
            "variant_NMD",
            "variant_POSITIVE_TRAIN_SITE",
            "variant_Pathway(BioCarta)_short",
            "variant_Polyphen2_HDIV_pred",
            "variant_Polyphen2_HVAR_pred",
            "variant_QD",
            "variant_RAW_MQ",
            "variant_RVIS",
            "variant_RVIS_percentile",
            "variant_ReadPosRankSum",
            "variant_Regulome_dbSNP141",
            "variant_SIFT_pred",
            "variant_SOR",
            "variant_Uniprot_aapos_Polyphen2",
            "variant_Uniprot_id_Polyphen2",
            "variant_VQSLOD",
            "variant_VariantType",
            "variant_ZFIN_zebrafish_gene",
            "variant_ZFIN_zebrafish_phenotype_tag",
            "variant_culprit",
            "variant_dbSNPBuildID",
            "variant_phastCons20way_mammalian",
            "variant_phyloP20way_mammalian",
            "variant_repeats",
        ]
        # Verified with bcftools view -H | grep INFO
        assert len(info_vars) == 140
        standard_vars = [
            "variant_filter",
            "variant_contig",
            "variant_position",
            "variant_length",
            "variant_allele",
            "variant_id",
            "variant_id_mask",
            "variant_quality",
            "contig_id",
            "contig_length",
            "filter_id",
            "filter_description",
            "region_index",
            "sample_id",
        ]
        assert sorted(list(ds)) == sorted(info_vars + standard_vars)

    def test_variant_ANN(self, ds):
        variant_ANN = [
            "A|intergenic_region|MODIFIER|DEFB125|ENSG00000178591|intergenic_region|ENSG00000178591|||n.60070G>A||||||",
            "C|intergenic_region|MODIFIER|DEFB125|ENSG00000178591|intergenic_region|ENSG00000178591|||n.60083T>C||||||",
            "C|intergenic_region|MODIFIER|DEFB125|ENSG00000178591|intergenic_region|ENSG00000178591|||n.60114T>C||||||",
            "G|intergenic_region|MODIFIER|DEFB125|ENSG00000178591|intergenic_region|ENSG00000178591|||n.60116A>G||||||",
            "C|intergenic_region|MODIFIER|DEFB125|ENSG00000178591|intergenic_region|ENSG00000178591|||n.60137T>C||||||",
            "A|intergenic_region|MODIFIER|DEFB125|ENSG00000178591|intergenic_region|ENSG00000178591|||n.60138T>A||||||",
            "T|intergenic_region|MODIFIER|DEFB125|ENSG00000178591|intergenic_region|ENSG00000178591|||n.60149C>T||||||",
            "G|intergenic_region|MODIFIER|DEFB125|ENSG00000178591|intergenic_region|ENSG00000178591|||n.60181A>G||||||",
            "G|intergenic_region|MODIFIER|DEFB125|ENSG00000178591|intergenic_region|ENSG00000178591|||n.60183A>G||||||",
            "A|intergenic_region|MODIFIER|DEFB125|ENSG00000178591|intergenic_region|ENSG00000178591|||n.60254C>A||||||",
            "T|intergenic_region|MODIFIER|DEFB125|ENSG00000178591|intergenic_region|ENSG00000178591|||n.60281_60285delTTCCA||||||",
            "TTTCCA|intergenic_region|MODIFIER|DEFB125|ENSG00000178591|intergenic_region|ENSG00000178591|||n.60280_60281insTTCCA||||||",
            "G|intergenic_region|MODIFIER|DEFB125|ENSG00000178591|intergenic_region|ENSG00000178591|||n.60286T>G||||||",
            "T|intergenic_region|MODIFIER|DEFB125|ENSG00000178591|intergenic_region|ENSG00000178591|||n.60287_60291delTCCAG||||||",
            "T|intergenic_region|MODIFIER|DEFB125|ENSG00000178591|intergenic_region|ENSG00000178591|||n.60291G>T||||||",
            "GTCCAT|intergenic_region|MODIFIER|DEFB125|ENSG00000178591|intergenic_region|ENSG00000178591|||n.60291_60292insTCCAT||||||",
            "G|intergenic_region|MODIFIER|DEFB125|ENSG00000178591|intergenic_region|ENSG00000178591|||n.60292_60301delTCCATTCCAT||||||",
            "G|intergenic_region|MODIFIER|DEFB125|ENSG00000178591|intergenic_region|ENSG00000178591|||n.60292_60296delTCCAT||||||",
            "GTCCATTCCAT|intergenic_region|MODIFIER|DEFB125|ENSG00000178591|intergenic_region|ENSG00000178591|||n.60291_60292insTCCATTCCAT||||||",
            "G|intergenic_region|MODIFIER|DEFB125|ENSG00000178591|intergenic_region|ENSG00000178591|||n.60329C>G||||||",
            "C|intergenic_region|MODIFIER|DEFB125|ENSG00000178591|intergenic_region|ENSG00000178591|||n.60331T>C||||||",
        ]
        nt.assert_array_equal(ds.variant_ANN.values, variant_ANN)

    def test_variant_MLEAF(self, ds):
        # fixes https://github.com/sgkit-dev/bio2zarr/issues/353
        assert ds.variant_MLEAF.dims == ("variants", "alt_alleles")
        assert ds.variant_MLEAF.shape == (21, 1)


class TestGeneratedFieldsExample:
    data_path = "tests/data/vcf/field_type_combos.vcf.gz"

    @pytest.fixture(scope="class")
    def ds(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("data") / "vcf.zarr"
        vcf_mod.convert([self.data_path], out)
        return sg.load_dataset(out)

    def test_info_string1(self, ds):
        values = ds["variant_IS1"].values
        non_missing = values[values != "."]
        nt.assert_array_equal(non_missing, ["bc"])

    def test_info_char1(self, ds):
        values = ds["variant_IC1"].values
        non_missing = values[values != "."]
        nt.assert_array_equal(non_missing, "f")

    def test_info_string2(self, ds):
        values = ds["variant_IS2"].values
        missing = np.all(values == ".", axis=1)
        non_missing_rows = values[~missing]
        nt.assert_array_equal(
            non_missing_rows, [["hij", "d"], [".", "d"], ["hij", "."]]
        )

    # FIXME can't figure out how to do the row masking properly here
    # def test_format_string1(self, ds):
    #     values = ds["call_FS1"].values
    #     missing = np.all(values == ".", axis=1)
    #     non_missing_rows = values[~missing]
    #     print(non_missing_rows)
    #     # nt.assert_array_equal(non_missing_rows, [["bc"], ["."]])

    # def test_format_string2(self, ds):
    #     values = ds["call_FS2"].values
    #     missing = np.all(values == ".", axis=1)
    #     non_missing_rows = values[~missing]
    #     non_missing = [v for v in pcvcf["FORMAT/FS2"].values if v is not None]
    #     nt.assert_array_equal(non_missing[0], [["bc", "op"], [".", "op"]])
    #     nt.assert_array_equal(non_missing[1], [["bc", "."], [".", "."]])


class TestSplitFileErrors:
    def test_entirely_incompatible(self, tmp_path):
        path = "tests/data/vcf/"
        with pytest.raises(ValueError, match="Incompatible"):
            vcf_mod.explode_init(
                tmp_path / "if", [path + "sample.vcf.gz", path + "1kg_2020_chrM.bcf"]
            )

    def test_duplicate_paths(self, tmp_path):
        path = "tests/data/vcf/"
        with pytest.raises(ValueError, match="Duplicate"):
            vcf_mod.explode_init(tmp_path / "if", [path + "sample.vcf.gz"] * 2)


@pytest.mark.parametrize(
    "name",
    [
        "sample.vcf.gz",
        "sample_old_tabix.vcf.gz",
        "sample_no_genotypes.vcf.gz",
        "sample_no_genotypes_with_gt_header.vcf.gz",
        "1kg_2020_chrM.vcf.gz",
        "field_type_combos.vcf.gz",
        "out_of_order_contigs.vcf.gz",
        "chr_m_indels.vcf.gz",
        "issue_251.vcf.gz",
    ],
)
def test_by_validating(name, tmp_path):
    path = f"tests/data/vcf/{name}"
    out = tmp_path / "test.zarr"
    vcf_mod.convert([path], out, worker_processes=0)
    vcz_verification.verify(path, out)


@pytest.mark.parametrize(
    ("source", "suffix", "files"),
    [
        ("sample.vcf.gz", "3.split", ["19:1-.vcf.gz", "20.vcf.gz", "X.vcf.gz"]),
        ("sample.vcf.gz", "3.split", ["20.vcf.gz", "19:1-.vcf.gz", "X.vcf.gz"]),
        ("out_of_order_contigs.vcf.gz", "2.split", ["A.vcf.gz", "B:1-.vcf.gz"]),
        ("out_of_order_contigs.vcf.gz", "2.split", ["A.bcf", "B:1-.bcf"]),
        ("out_of_order_contigs.vcf.gz", "2.split", ["A.vcf.gz", "B:1-.bcf"]),
    ],
)
def test_by_validating_split(source, suffix, files, tmp_path):
    source_path = f"tests/data/vcf/{source}"
    split_files = [f"{source_path}.{suffix}/{f}" for f in files]
    out = tmp_path / "test.zarr"
    vcf_mod.convert(split_files, out, worker_processes=0)
    vcz_verification.verify(source_path, out)


def test_split_explode(tmp_path):
    paths = [
        "tests/data/vcf/sample.vcf.gz.3.split/19:1-.vcf.gz",
        "tests/data/vcf/sample.vcf.gz.3.split/20.vcf.gz",
        "tests/data/vcf/sample.vcf.gz.3.split/X.vcf.gz",
    ]
    out = tmp_path / "test.explode"
    work_summary = vcf_mod.explode_init(out, paths, target_num_partitions=15)
    assert work_summary.num_partitions == 3

    with pytest.raises(FileNotFoundError):
        pcvcf = vcf_mod.IntermediateColumnarFormat(out)

    for j in range(work_summary.num_partitions):
        vcf_mod.explode_partition(out, j)
    vcf_mod.explode_finalise(out)
    pcvcf = vcf_mod.IntermediateColumnarFormat(out)
    summary_d = pcvcf.fields["POS"].vcf_field.summary.asdict()
    # The compressed size can vary with different numcodecs versions
    assert summary_d["compressed_size"] in [571, 573, 587]
    del summary_d["compressed_size"]
    assert summary_d == {
        "num_chunks": 3,
        "uncompressed_size": 1008,
        "max_number": 1,
        "max_value": 1235237,
        "min_value": 10,
    }
    vcf_mod.encode(out, tmp_path / "test.zarr")
    vcz_verification.verify("tests/data/vcf/sample.vcf.gz", tmp_path / "test.zarr")


def test_missing_filter(tmp_path):
    path = "tests/data/vcf/sample_missing_filter.vcf.gz"
    zarr_path = tmp_path / "zarr"
    with pytest.raises(ValueError, match="Filter 'q10' was not defined in the header"):
        vcf_mod.convert([path], zarr_path)


class TestOutOfOrderFields:
    # Mixing on purpose
    data_path1 = "tests/data/vcf/out_of_order_fields/input2.bcf"
    data_path2 = "tests/data/vcf/out_of_order_fields/input1.bcf"

    @pytest.fixture(scope="class")
    def ds(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("data") / "ooo_example.vcf.zarr"
        vcf_mod.convert([self.data_path1, self.data_path2], out)
        return sg.load_dataset(out)

    def test_filters(self, ds):
        nt.assert_array_equal(ds["filter_id"], ["PASS", "FAIL"])
        nt.assert_array_equal(
            ds["variant_filter"],
            [
                [True, False],
                [False, True],
                [True, False],
            ],
        )

    def test_source(self, ds):
        assert ds.attrs["source"] == f"bio2zarr-{provenance.__version__}"

    def test_contigs(self, ds):
        nt.assert_array_equal(ds["contig_id"], ["chr20", "chr21"])
        nt.assert_array_equal(ds["contig_length"], [64444167.0, 46709983.0])
        nt.assert_array_equal(ds["variant_contig"], [0, 1, 1])

    def test_position(self, ds):
        nt.assert_array_equal(ds["variant_position"], [63971, 64506, 64507])

    def test_length(self, ds):
        nt.assert_array_equal(ds["variant_length"], [11, 1, 1])

    def test_info_fields(self, ds):
        nt.assert_array_equal(
            ds["variant_QNAME"],
            ["cluster19_000000F", ".", "cluster19_000000F"],
        )
        nt.assert_array_equal(ds["variant_QSTART"], [25698928, 25698928, -1])

    def test_allele(self, ds):
        nt.assert_array_equal(
            ds["variant_allele"].values.tolist(),
            [["TTCCATTCCAC", "T"], ["C", "CTCCAT"], ["G", "A"]],
        )
        assert ds["variant_allele"].dtype == "O"

    def test_call_DPs(self, ds):
        nt.assert_array_equal(ds["call_DP"], [[5], [-1], [5]])
        nt.assert_array_equal(ds["call_DP2"], [[1], [1], [-1]])
