import pytest
from cyvcf2 import VCF

from bio2zarr.csi import read_csi
from bio2zarr.tbi import read_tabix
from bio2zarr.vcf_partition import get_csi_path
from bio2zarr.vcf_partition import get_tabix_path

from .utils import count_variants, path_for_test


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
