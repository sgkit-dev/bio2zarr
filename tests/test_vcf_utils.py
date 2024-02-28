import pathlib

import pytest
from cyvcf2 import VCF
import numpy as np

from bio2zarr import vcf_utils

from bio2zarr.vcf_utils import read_csi
from bio2zarr.vcf_utils import read_tabix
from bio2zarr.vcf_utils import get_csi_path
from bio2zarr.vcf_utils import get_tabix_path
from bio2zarr.vcf_utils import partition_into_regions


from .utils import count_variants, path_for_test

data_path = pathlib.Path("tests/data/vcf/")


# values computed using bcftools index -s
@pytest.mark.parametrize(
    ["index_file", "expected"],
    [
        ("sample.vcf.gz.tbi", {"19": 2, "20": 6, "X": 1}),
        ("sample.bcf.csi", {"19": 2, "20": 6, "X": 1}),
        ("sample_no_genotypes.vcf.gz.csi", {"19": 2, "20": 6, "X": 1}),
        ("CEUTrio.20.21.gatk3.4.g.vcf.bgz.tbi", {"20": 3450, "21": 16460}),
        ("CEUTrio.20.21.gatk3.4.g.bcf.csi", {"20": 3450, "21": 16460}),
        ("1kg_2020_chrM.vcf.gz.tbi", {"chrM": 23}),
        ("1kg_2020_chrM.vcf.gz.csi", {"chrM": 23}),
        ("1kg_2020_chrM.bcf.csi", {"chrM": 23}),
        ("1kg_2020_chr20_annotations.bcf.csi", {"chr20": 21}),
        ("NA12878.prod.chr20snippet.g.vcf.gz.tbi", {"20": 301778}),
        ("multi_contig.vcf.gz.tbi", {str(j): 933 for j in range(5)}),
    ],
)
def test_index_record_count(index_file, expected):
    vcf_path = data_path / (".".join(list(index_file.split("."))[:-1]))
    indexed_vcf = vcf_utils.IndexedVcf(vcf_path, data_path / index_file)
    assert indexed_vcf.contig_record_counts() == expected


@pytest.mark.parametrize(
    ["index_file", "expected"],
    [
        ("sample.vcf.gz.tbi", ["19:1-", "20", "X"]),
        ("sample.bcf.csi", ["19:1-", "20", "X"]),
        ("sample_no_genotypes.vcf.gz.csi", ["19:1-", "20", "X"]),
        ("CEUTrio.20.21.gatk3.4.g.vcf.bgz.tbi", ["20:1-", "21"]),
        ("CEUTrio.20.21.gatk3.4.g.bcf.csi", ["20:1-", "21"]),
        ("1kg_2020_chrM.vcf.gz.tbi", ["chrM:1-"]),
        ("1kg_2020_chrM.vcf.gz.csi", ["chrM:1-"]),
        ("1kg_2020_chrM.bcf.csi", ["chrM:1-"]),
        ("1kg_2020_chr20_annotations.bcf.csi", ["chr20:49153-"]),
        ("NA12878.prod.chr20snippet.g.vcf.gz.tbi", ["20:1-"]),
        ("multi_contig.vcf.gz.tbi", ["0:1-"] + [str(j) for j in range(1, 5)]),
    ],
)
def test_partition_into_one_part(index_file, expected):
    vcf_path = data_path / (".".join(list(index_file.split("."))[:-1]))
    indexed_vcf = vcf_utils.IndexedVcf(vcf_path, data_path / index_file)
    regions = indexed_vcf.partition_into_regions(num_parts=1)
    assert all(isinstance(r, vcf_utils.Region) for r in regions)
    assert [str(r) for r in regions] == expected


def test_tabix_multi_chrom_bug():
    index_file = "multi_contig.vcf.gz.tbi"
    vcf_path = data_path / (".".join(list(index_file.split("."))[:-1]))
    indexed_vcf = vcf_utils.IndexedVcf(vcf_path, data_path / index_file)
    regions = indexed_vcf.partition_into_regions(num_parts=10)
    # An earlier version of the code returned this, i.e. with a duplicate
    # for 4 with end coord of 0
    # ["0:1-", "1", "2", "3", "4:1-0", "4:1-"]
    expected = ["0:1-", "1", "2", "3", "4:1-"]
    assert [str(r) for r in regions] == expected


class TestCsiIndex:
    @pytest.mark.parametrize(
        "filename",
        ["CEUTrio.20.21.gatk3.4.g.vcf.bgz", "CEUTrio.20.21.gatk3.4.g.vcf.bgz.tbi"],
    )
    def test_invalid_csi(self, filename):
        with pytest.raises(ValueError, match=r"File not in CSI format."):
            read_csi(data_path / filename)


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
            read_tabix(data_path / filename)


class TestPartitionIntoRegions:
    @pytest.mark.parametrize(
        "vcf_file",
        [
            "CEUTrio.20.21.gatk3.4.g.bcf",
            "CEUTrio.20.21.gatk3.4.g.vcf.bgz",
            "NA12878.prod.chr20snippet.g.vcf.gz",
        ],
    )
    def test_num_parts(self, vcf_file):
        vcf_path = data_path / vcf_file
        regions = partition_into_regions(vcf_path, num_parts=4)

        assert regions is not None
        part_variant_counts = [count_variants(vcf_path, region) for region in regions]
        total_variants = count_variants(vcf_path)

        assert sum(part_variant_counts) == total_variants

    def test_num_parts_large(self):
        vcf_path = data_path / "CEUTrio.20.21.gatk3.4.g.vcf.bgz"

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
    def test_target_part_size(self, target_part_size):
        vcf_path = data_path / "CEUTrio.20.21.gatk3.4.g.vcf.bgz"

        regions = partition_into_regions(vcf_path, target_part_size=target_part_size)
        assert regions is not None
        assert len(regions) == 5

        part_variant_counts = [count_variants(vcf_path, region) for region in regions]
        assert part_variant_counts == [3450, 3869, 4525, 7041, 1025]
        total_variants = count_variants(vcf_path)

        assert sum(part_variant_counts) == total_variants

    def test_invalid_arguments(self):
        vcf_path = data_path / "CEUTrio.20.21.gatk3.4.g.vcf.bgz"

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

    @pytest.mark.skip("TODO")
    def test_missing_index(self, temp_path):
        vcf_path = data_path / "CEUTrio.20.21.gatk3.4.g.vcf.bgz"
        with pytest.raises(ValueError, match=r"Cannot find .tbi or .csi file."):
            partition_into_regions(vcf_path, num_parts=2)

        bogus_index_path = path_for_test(
            shared_datadir, "CEUTrio.20.21.gatk3.4.noindex.g.vcf.bgz.index", True
        )
        with pytest.raises(
            ValueError, match=r"Only .tbi or .csi indexes are supported."
        ):
            partition_into_regions(vcf_path, index_path=bogus_index_path, num_parts=2)
