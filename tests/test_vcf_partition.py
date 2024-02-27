import pytest

from bio2zarr.vcf_partition import partition_into_regions

from .utils import count_variants, path_for_test


@pytest.mark.parametrize(
    "vcf_file",
    [
        "CEUTrio.20.21.gatk3.4.g.bcf",
        "CEUTrio.20.21.gatk3.4.g.vcf.bgz",
        "NA12878.prod.chr20snippet.g.vcf.gz",
    ],
)
def test_partition_into_regions__num_parts(shared_datadir, vcf_file):
    vcf_path = path_for_test(shared_datadir, vcf_file, True)

    regions = partition_into_regions(vcf_path, num_parts=4)

    assert regions is not None
    part_variant_counts = [count_variants(vcf_path, region) for region in regions]
    total_variants = count_variants(vcf_path)

    assert sum(part_variant_counts) == total_variants


def test_partition_into_regions__num_parts_large(shared_datadir):
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
def test_partition_into_regions__target_part_size(shared_datadir, target_part_size):
    vcf_path = path_for_test(shared_datadir, "CEUTrio.20.21.gatk3.4.g.vcf.bgz")

    regions = partition_into_regions(vcf_path, target_part_size=target_part_size)
    assert regions is not None
    assert len(regions) == 5

    part_variant_counts = [count_variants(vcf_path, region) for region in regions]
    total_variants = count_variants(vcf_path)

    assert sum(part_variant_counts) == total_variants


def test_partition_into_regions__invalid_arguments(shared_datadir):
    vcf_path = path_for_test(shared_datadir, "CEUTrio.20.21.gatk3.4.g.vcf.bgz")

    with pytest.raises(
        ValueError, match=r"One of num_parts or target_part_size must be specified"
    ):
        partition_into_regions(vcf_path)

    with pytest.raises(
        ValueError, match=r"Only one of num_parts or target_part_size may be specified"
    ):
        partition_into_regions(vcf_path, num_parts=4, target_part_size=100_000)

    with pytest.raises(ValueError, match=r"num_parts must be positive"):
        partition_into_regions(vcf_path, num_parts=0)

    with pytest.raises(ValueError, match=r"target_part_size must be positive"):
        partition_into_regions(vcf_path, target_part_size=0)


def test_partition_into_regions__one_part(shared_datadir):
    vcf_path = path_for_test(shared_datadir, "CEUTrio.20.21.gatk3.4.g.vcf.bgz")
    parts = partition_into_regions(vcf_path, num_parts=1)
    assert parts == ["20:1-", "21"]


def test_partition_into_regions__missing_index(shared_datadir):
    vcf_path = path_for_test(
        shared_datadir, "CEUTrio.20.21.gatk3.4.noindex.g.vcf.bgz", True
    )
    with pytest.raises(ValueError, match=r"Cannot find .tbi or .csi file."):
        partition_into_regions(vcf_path, num_parts=2)

    bogus_index_path = path_for_test(
        shared_datadir, "CEUTrio.20.21.gatk3.4.noindex.g.vcf.bgz.index", True
    )
    with pytest.raises(ValueError, match=r"Only .tbi or .csi indexes are supported."):
        partition_into_regions(vcf_path, index_path=bogus_index_path, num_parts=2)
