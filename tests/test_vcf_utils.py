import pytest
from cyvcf2 import VCF

from bio2zarr.vcf_utils import read_csi
from bio2zarr.vcf_utils import read_tabix
from bio2zarr.vcf_utils import get_csi_path
from bio2zarr.vcf_utils import get_tabix_path
from bio2zarr.vcf_utils import partition_into_regions


from .utils import count_variants, path_for_test


class TestCEUTrio2021VcfExample:
    data_path = "tests/data/vcf/CEUTrio.20.21.gatk3.4.g.vcf.bgz"

    @pytest.fixture(scope="class")
    def index(self):
        tabix_path = get_tabix_path(self.data_path)
        return read_tabix(tabix_path)

    def test_record_counts(self, index):
        # print(index.sequence_names)
        print(index.record_counts)
        # for i, contig in enumerate(tabix.sequence_names):
        #     assert tabix.record_counts[i] == count_variants(vcf_path, contig)

    # def test_one_region(self, index):
    #     parts = partition_into_regions(self.data_path, num_parts=1)
    #     assert parts == ["20:1-", "21"]


class TestCEUTrio2021BcfExample(TestCEUTrio2021VcfExample):
    data_path = "tests/data/vcf/CEUTrio.20.21.gatk3.4.g.bcf"

    @pytest.fixture(scope="class")
    def index(self):
        csi_path = get_csi_path(self.data_path)
        return read_csi(csi_path)


class TestCsiIndex:
    @pytest.mark.parametrize(
        "vcf_file",
        [
            "CEUTrio.20.21.gatk3.4.csi.g.vcf.bgz",
        ],
    )
    def test_record_counts(self, shared_datadir, vcf_file):
        # Check record counts in csi with actual count of VCF
        vcf_path = path_for_test(shared_datadir, vcf_file, True)
        csi_path = get_csi_path(vcf_path)
        assert csi_path is not None
        csi = read_csi(csi_path)

        for i, contig in enumerate(VCF(vcf_path).seqnames):
            assert csi.record_counts[i] == count_variants(vcf_path, contig)

    @pytest.mark.parametrize(
        "file",
        ["CEUTrio.20.21.gatk3.4.g.vcf.bgz", "CEUTrio.20.21.gatk3.4.g.vcf.bgz.tbi"],
    )
    def test_invalid_csi(self, shared_datadir, file):
        with pytest.raises(ValueError, match=r"File not in CSI format."):
            read_csi(path_for_test(shared_datadir, file, True))

    @pytest.mark.parametrize(
        "file",
        ["CEUTrio.20.21.gatk3.4.g.vcf.bgz", "CEUTrio.20.21.gatk3.4.g.vcf.bgz.tbi"],
    )
    def test_invalid_csi(self, shared_datadir, file):
        with pytest.raises(ValueError, match=r"File not in CSI format."):
            read_csi(path_for_test(shared_datadir, file, True))


class TestTabixIndex:
    @pytest.mark.parametrize(
        "vcf_file",
        [
            "CEUTrio.20.21.gatk3.4.g.vcf.bgz",
        ],
    )
    def test_record_counts(self, shared_datadir, vcf_file):
        # Check record counts in tabix with actual count of VCF
        vcf_path = path_for_test(shared_datadir, vcf_file, True)
        tabix_path = get_tabix_path(vcf_path)
        assert tabix_path is not None
        tabix = read_tabix(tabix_path)

        for i, contig in enumerate(tabix.sequence_names):
            assert tabix.record_counts[i] == count_variants(vcf_path, contig)

    @pytest.mark.parametrize(
        "file",
        ["CEUTrio.20.21.gatk3.4.g.vcf.bgz", "CEUTrio.20.21.gatk3.4.csi.g.vcf.bgz.csi"],
    )
    def test_read_tabix__invalid_tbi(self, shared_datadir, file):
        with pytest.raises(ValueError, match=r"File not in Tabix format."):
            read_tabix(path_for_test(shared_datadir, file, True))


class TestPartitionIntoRegions:
    @pytest.mark.parametrize(
        "vcf_file",
        [
            "CEUTrio.20.21.gatk3.4.g.bcf",
            "CEUTrio.20.21.gatk3.4.g.vcf.bgz",
            "NA12878.prod.chr20snippet.g.vcf.gz",
        ],
    )
    def test_num_parts(self, shared_datadir, vcf_file):
        vcf_path = path_for_test(shared_datadir, vcf_file, True)

        regions = partition_into_regions(vcf_path, num_parts=4)

        assert regions is not None
        part_variant_counts = [count_variants(vcf_path, region) for region in regions]
        total_variants = count_variants(vcf_path)

        assert sum(part_variant_counts) == total_variants

    def test_num_parts_large(self, shared_datadir):
        vcf_path = path_for_test(shared_datadir, "CEUTrio.20.21.gatk3.4.g.vcf.bgz")

        regions = partition_into_regions(vcf_path, num_parts=100)
        assert regions is not None
        assert len(regions) == 18

        part_variant_counts = [count_variants(vcf_path, region) for region in regions]
        total_variants = count_variants(vcf_path)

        assert sum(part_variant_counts) == total_variants

    @pytest.mark.parametrize(
        "target_part_size",
        [
            100_000,
            "100KB",
            "100 kB",
        ],
    )
    def test_target_part_size(self, shared_datadir, target_part_size):
        vcf_path = path_for_test(shared_datadir, "CEUTrio.20.21.gatk3.4.g.vcf.bgz")

        regions = partition_into_regions(vcf_path, target_part_size=target_part_size)
        assert regions is not None
        assert len(regions) == 5

        part_variant_counts = [count_variants(vcf_path, region) for region in regions]
        total_variants = count_variants(vcf_path)

        assert sum(part_variant_counts) == total_variants

    def test_invalid_arguments(self, shared_datadir):
        vcf_path = path_for_test(shared_datadir, "CEUTrio.20.21.gatk3.4.g.vcf.bgz")

        with pytest.raises(
            ValueError, match=r"One of num_parts or target_part_size must be specified"
        ):
            partition_into_regions(vcf_path)

        with pytest.raises(
            ValueError,
            match=r"Only one of num_parts or target_part_size may be specified",
        ):
            partition_into_regions(vcf_path, num_parts=4, target_part_size=100_000)

        with pytest.raises(ValueError, match=r"num_parts must be positive"):
            partition_into_regions(vcf_path, num_parts=0)

        with pytest.raises(ValueError, match=r"target_part_size must be positive"):
            partition_into_regions(vcf_path, target_part_size=0)

    def test_one_part(self, shared_datadir):
        vcf_path = path_for_test(shared_datadir, "CEUTrio.20.21.gatk3.4.g.vcf.bgz")
        parts = partition_into_regions(vcf_path, num_parts=1)
        assert parts == ["20:1-", "21"]

    def test_missing_index(self, shared_datadir):
        vcf_path = path_for_test(
            shared_datadir, "CEUTrio.20.21.gatk3.4.noindex.g.vcf.bgz", True
        )
        with pytest.raises(ValueError, match=r"Cannot find .tbi or .csi file."):
            partition_into_regions(vcf_path, num_parts=2)

        bogus_index_path = path_for_test(
            shared_datadir, "CEUTrio.20.21.gatk3.4.noindex.g.vcf.bgz.index", True
        )
        with pytest.raises(
            ValueError, match=r"Only .tbi or .csi indexes are supported."
        ):
            partition_into_regions(vcf_path, index_path=bogus_index_path, num_parts=2)
