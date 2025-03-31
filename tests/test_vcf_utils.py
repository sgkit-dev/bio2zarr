import pathlib
import shutil

import numpy as np
import pytest

from bio2zarr import vcf_utils
from bio2zarr.vcf_utils import RECORD_COUNT_UNKNOWN

data_path = pathlib.Path("tests/data/vcf/")


def assert_part_counts_non_zero(part_counts, index_file):
    assert np.all(part_counts > 0)


class TestVcfFile:
    def get_instance(self, index_file):
        vcf_path = data_path / (".".join(list(index_file.split("."))[:-1]))
        return vcf_utils.VcfFile(vcf_path, data_path / index_file)

    def test_context_manager_success(self):
        # Nominal case
        with vcf_utils.VcfFile(data_path / "sample.bcf") as iv:
            assert iv.vcf is not None
        assert iv.vcf is None

    def test_context_manager_error(self):
        with pytest.raises(FileNotFoundError, match="no-such-file"):
            with vcf_utils.VcfFile(data_path / "no-such-file.bcf"):
                pass

    def test_indels_filtered(self):
        with vcf_utils.VcfFile(data_path / "chr_m_indels.vcf.gz") as vfile:
            # Hand-picked example that results in filtering
            region = vcf_utils.Region("chrM", 300, 314)
            pos = [var.POS for var in vfile.variants(region)]
            assert pos == [307, 308, 309, 312, 313, 314]

    # values computed using bcftools index -s
    @pytest.mark.parametrize(
        ("index_file", "expected"),
        [
            ("sample.vcf.gz.tbi", {"19": 2, "20": 6, "X": 1}),
            (
                "sample_old_tabix.vcf.gz.tbi",
                {
                    "19": RECORD_COUNT_UNKNOWN,
                    "20": RECORD_COUNT_UNKNOWN,
                    "X": RECORD_COUNT_UNKNOWN,
                },
            ),
            ("sample.bcf.csi", {"19": 2, "20": 6, "X": 1}),
            ("sample_extra_contig.vcf.gz.csi", {"19": 2, "20": 6, "X": 1}),
            ("sample_extra_contig.bcf.csi", {"19": 2, "20": 6, "X": 1}),
            ("sample_no_genotypes.vcf.gz.csi", {"19": 2, "20": 6, "X": 1}),
            ("CEUTrio.20.21.gatk3.4.g.vcf.bgz.tbi", {"20": 3450, "21": 16460}),
            (
                "CEUTrio.20.21.gatk3.4.g.old_tabix.vcf.bgz.tbi",
                {"20": RECORD_COUNT_UNKNOWN, "21": RECORD_COUNT_UNKNOWN},
            ),
            ("CEUTrio.20.21.gatk3.4.g.bcf.csi", {"20": 3450, "21": 16460}),
            ("1kg_2020_chrM.vcf.gz.tbi", {"chrM": 23}),
            ("1kg_2020_chrM.vcf.gz.csi", {"chrM": 23}),
            ("1kg_2020_chrM.bcf.csi", {"chrM": 23}),
            ("1kg_2020_chr20_annotations.bcf.csi", {"chr20": 21}),
            ("NA12878.prod.chr20snippet.g.vcf.gz.tbi", {"20": 301778}),
            ("multi_contig.vcf.gz.tbi", {str(j): 933 for j in range(5)}),
            ("chr_m_indels.vcf.gz.csi", {"chrM": 155}),
        ],
    )
    def test_contig_record_counts(self, index_file, expected):
        indexed_vcf = self.get_instance(index_file)
        assert indexed_vcf.contig_record_counts() == expected

    @pytest.mark.parametrize(
        ("index_file", "expected"),
        [
            ("sample.vcf.gz.tbi", ["19:111-", "20:14370-", "X:10-"]),
            ("sample_old_tabix.vcf.gz.tbi", ["19:111-", "20:14370-", "X:10-"]),
            ("sample.bcf.csi", ["19:111-", "20:14370-", "X:10-"]),
            ("sample_extra_contig.bcf.csi", ["19:111-", "20:14370-", "X:10-"]),
            ("sample_extra_contig.vcf.gz.csi", ["19:111-", "20:14370-", "X:10-"]),
            ("sample_no_genotypes.vcf.gz.csi", ["19:111-", "20:14370-", "X:10-"]),
            ("CEUTrio.20.21.gatk3.4.g.vcf.bgz.tbi", ["20:1-", "21:1-"]),
            ("CEUTrio.20.21.gatk3.4.g.old_tabix.vcf.bgz.tbi", ["20:1-", "21:1-"]),
            ("CEUTrio.20.21.gatk3.4.g.bcf.csi", ["20:1-", "21:1-"]),
            ("1kg_2020_chrM.vcf.gz.tbi", ["chrM:26-"]),
            ("1kg_2020_chrM.vcf.gz.csi", ["chrM:26-"]),
            ("1kg_2020_chrM.bcf.csi", ["chrM:26-"]),
            ("1kg_2020_chr20_annotations.bcf.csi", ["chr20:60070-"]),
            ("NA12878.prod.chr20snippet.g.vcf.gz.tbi", ["20:60001-"]),
            ("multi_contig.vcf.gz.tbi", [f"{j}:1-" for j in range(5)]),
            ("chr_m_indels.vcf.gz.csi", ["chrM:26-"]),
        ],
    )
    def test_partition_into_one_part(self, index_file, expected):
        indexed_vcf = self.get_instance(index_file)
        regions = list(indexed_vcf.partition_into_regions(num_parts=1))
        assert all(isinstance(r, vcf_utils.Region) for r in regions)
        assert [str(r) for r in regions] == expected

    @pytest.mark.parametrize(
        ("index_file", "num_expected", "total_records"),
        [
            ("sample.vcf.gz.tbi", 3, 9),
            ("sample_old_tabix.vcf.gz.tbi", 3, 9),
            ("sample.bcf.csi", 3, 9),
            ("sample_no_genotypes.vcf.gz.csi", 3, 9),
            ("CEUTrio.20.21.gatk3.4.g.vcf.bgz.tbi", 17, 19910),
            ("CEUTrio.20.21.gatk3.4.g.old_tabix.vcf.bgz.tbi", 17, 19910),
            ("CEUTrio.20.21.gatk3.4.g.bcf.csi", 3, 19910),
            ("1kg_2020_chrM.vcf.gz.tbi", 1, 23),
            ("1kg_2020_chrM.vcf.gz.csi", 1, 23),
            ("1kg_2020_chrM.bcf.csi", 1, 23),
            ("1kg_2020_chr20_annotations.bcf.csi", 1, 21),
            ("NA12878.prod.chr20snippet.g.vcf.gz.tbi", 59, 301778),
            ("multi_contig.vcf.gz.tbi", 5, 5 * 933),
            ("chr_m_indels.vcf.gz.csi", 1, 155),
        ],
    )
    def test_partition_into_max_parts(self, index_file, num_expected, total_records):
        indexed_vcf = self.get_instance(index_file)
        regions = list(indexed_vcf.partition_into_regions(num_parts=1000))
        assert all(isinstance(r, vcf_utils.Region) for r in regions)
        # print(regions)
        assert len(regions) == num_expected
        part_variant_counts = np.array(
            [indexed_vcf.count_variants(region) for region in regions]
        )
        assert np.sum(part_variant_counts) == total_records
        assert_part_counts_non_zero(part_variant_counts, index_file)

    @pytest.mark.parametrize(
        ("index_file", "total_records"),
        [
            ("sample.vcf.gz.tbi", 9),
            ("sample_old_tabix.vcf.gz.tbi", 9),
            ("sample.bcf.csi", 9),
            ("sample_no_genotypes.vcf.gz.csi", 9),
            ("CEUTrio.20.21.gatk3.4.g.vcf.bgz.tbi", 19910),
            ("CEUTrio.20.21.gatk3.4.g.old_tabix.vcf.bgz.tbi", 19910),
            ("CEUTrio.20.21.gatk3.4.g.bcf.csi", 19910),
            ("1kg_2020_chrM.vcf.gz.tbi", 23),
            ("1kg_2020_chrM.vcf.gz.csi", 23),
            ("1kg_2020_chrM.bcf.csi", 23),
            ("1kg_2020_chr20_annotations.bcf.csi", 21),
            ("NA12878.prod.chr20snippet.g.vcf.gz.tbi", 301778),
            ("multi_contig.vcf.gz.tbi", 5 * 933),
        ],
    )
    @pytest.mark.parametrize("num_parts", [2, 3, 4, 5, 16, 33])
    def test_partition_into_n_parts(self, index_file, total_records, num_parts):
        indexed_vcf = self.get_instance(index_file)
        regions = list(indexed_vcf.partition_into_regions(num_parts=num_parts))
        assert all(isinstance(r, vcf_utils.Region) for r in regions)
        part_variant_counts = np.array(
            [indexed_vcf.count_variants(region) for region in regions]
        )
        assert np.sum(part_variant_counts) == total_records
        assert_part_counts_non_zero(part_variant_counts, index_file)

    @pytest.mark.parametrize(
        ("vcf_file", "total_records"),
        [
            ("1kg_2020_chrM.vcf.gz", 23),
            ("1kg_2020_chr20_annotations.bcf", 21),
        ],
    )
    @pytest.mark.parametrize("num_parts", [1, 2, 3])
    def test_partition_into_n_parts_unindexed(
        self, tmp_path, vcf_file, total_records, num_parts
    ):
        copy_path = tmp_path / vcf_file
        shutil.copyfile(data_path / vcf_file, copy_path)
        indexed_vcf = vcf_utils.VcfFile(copy_path)
        regions = list(indexed_vcf.partition_into_regions(num_parts=num_parts))
        assert len(regions) == 1
        part_variant_counts = np.array(
            [indexed_vcf.count_variants(region) for region in regions]
        )
        assert np.sum(part_variant_counts) == total_records

    def test_tabix_multi_chrom_bug(self):
        indexed_vcf = self.get_instance("multi_contig.vcf.gz.tbi")
        regions = list(indexed_vcf.partition_into_regions(num_parts=10))
        # An earlier version of the code returned this, i.e. with a duplicate
        # for 4 with end coord of 0
        # ["0:1-", "1", "2", "3", "4:1-0", "4:1-"]
        expected = ["0:1-", "1:1-", "2:1-", "3:1-", "4:1-"]
        assert [str(r) for r in regions] == expected

    @pytest.mark.parametrize(
        "target_part_size",
        [
            100_000,
            "100KB",
            "100 kB",
        ],
    )
    @pytest.mark.parametrize(
        "filename",
        [
            "CEUTrio.20.21.gatk3.4.g.vcf.bgz.tbi",
            "CEUTrio.20.21.gatk3.4.g.old_tabix.vcf.bgz.tbi",
        ],
    )
    def test_target_part_size(self, target_part_size, filename):
        indexed_vcf = self.get_instance(filename)
        regions = list(
            indexed_vcf.partition_into_regions(target_part_size=target_part_size)
        )
        assert len(regions) == 5
        part_variant_counts = [indexed_vcf.count_variants(region) for region in regions]
        assert part_variant_counts == [3450, 3869, 4525, 7041, 1025]
        assert sum(part_variant_counts) == 19910

    def test_partition_invalid_arguments(self):
        indexed_vcf = self.get_instance("CEUTrio.20.21.gatk3.4.g.vcf.bgz.tbi")

        with pytest.raises(
            ValueError, match=r"One of num_parts or target_part_size must be specified"
        ):
            list(indexed_vcf.partition_into_regions())

        with pytest.raises(
            ValueError,
            match=r"Only one of num_parts or target_part_size may be specified",
        ):
            indexed_vcf.partition_into_regions(num_parts=4, target_part_size=100_000)

        with pytest.raises(ValueError, match=r"num_parts must be positive"):
            indexed_vcf.partition_into_regions(num_parts=0)

        with pytest.raises(ValueError, match=r"target_part_size must be positive"):
            indexed_vcf.partition_into_regions(target_part_size=0)

    @pytest.mark.parametrize("path", ["y", data_path / "xxx", "/x/y.csi"])
    def test_missing_index_file(self, path):
        with pytest.raises(FileNotFoundError, match="Specified index path"):
            vcf_utils.VcfFile(data_path / "sample.vcf.gz", path)

    def test_bad_index_format(self):
        vcf_file = data_path / "sample.vcf.gz"
        with pytest.raises(ValueError, match="Only .tbi or .csi indexes"):
            vcf_utils.VcfFile(vcf_file, vcf_file)

    @pytest.mark.parametrize(
        "filename",
        [
            "1kg_2020_chrM.vcf.gz",
            "1kg_2020_chrM.bcf",
            "1kg_2020_chr20_annotations.bcf",
            "chr_m_indels.vcf.gz",
            "NA12878.prod.chr20snippet.g.vcf.gz",
        ],
    )
    def test_unindexed_single_contig(self, tmp_path, filename):
        f1 = vcf_utils.VcfFile(data_path / filename)
        assert f1.index is not None
        copy_path = tmp_path / filename
        shutil.copyfile(data_path / filename, copy_path)
        f2 = vcf_utils.VcfFile(copy_path)
        assert f2.index is None
        crc1 = f1.contig_record_counts()
        assert len(crc1) == 1
        contig = next(iter(crc1.keys()))
        assert f2.contig_record_counts() == {contig: np.inf}
        region = vcf_utils.Region(contig)
        # The full variants returned by cyvcf2 don't compare for equality,
        # so just check the chrom/pos values
        v1 = [(v.CHROM, v.POS) for v in f1.variants(region)]
        v2 = [(v.CHROM, v.POS) for v in f2.variants(region)]
        assert v1 == v2

    @pytest.mark.parametrize(
        "filename",
        ["sample.vcf.gz", "sample.bcf", "multi_contig.vcf.gz"],
    )
    def test_unindexed_multi_contig(self, tmp_path, filename):
        copy_path = tmp_path / filename
        shutil.copyfile(data_path / filename, copy_path)
        f = vcf_utils.VcfFile(copy_path)
        with pytest.raises(ValueError, match="Multi-contig VCFs must be indexed"):
            list(f.variants())


class TestCsiIndex:
    @pytest.mark.parametrize(
        "filename",
        ["CEUTrio.20.21.gatk3.4.g.vcf.bgz", "CEUTrio.20.21.gatk3.4.g.vcf.bgz.tbi"],
    )
    def test_invalid_csi(self, filename):
        with pytest.raises(ValueError, match=r"File not in CSI format."):
            vcf_utils.read_csi(data_path / filename)


class TestTabixIndex:
    @pytest.mark.parametrize(
        "filename",
        [
            "CEUTrio.20.21.gatk3.4.g.vcf.bgz",
            "CEUTrio.20.21.gatk3.4.g.bcf.csi",
        ],
    )
    def test_invalid_tbi(self, filename):
        with pytest.raises(ValueError, match=r"File not in Tabix format."):
            vcf_utils.read_tabix(data_path / filename)
