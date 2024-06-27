import shutil

import pysam
from hypothesis import HealthCheck, given, note, settings
from hypothesis_vcf import vcf

from bio2zarr import vcf2zarr

# Make sure POS starts at 1, since CSI indexing doesn't seem to support zero-based coordinates
# (even when passing zerobased=True to pysam.tabix_index below)
@given(vcf_string=vcf(min_pos=1))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
def test_hypothesis_generated_vcf(tmp_path, vcf_string):
    note(f"vcf:\n{vcf_string}")

    path = tmp_path / "input.vcf"
    icf_path = tmp_path / "icf"
    zarr_path = tmp_path / "zarr"

    with open(path, "w") as f:
        f.write(vcf_string)

    # make sure outputs don't exist (from previous hypothesis example)
    shutil.rmtree(str(icf_path), ignore_errors=True)
    shutil.rmtree(str(zarr_path), ignore_errors=True)

    # create a tabix index for the VCF,
    # using CSI since POS can exceed range supported by TBI
    # (this also compresses the input file)
    pysam.tabix_index(str(path), preset="vcf", force=True, csi=True)

    # test that we can convert VCFs to Zarr without error
    vcf2zarr.convert([str(path) + ".gz"], zarr_path, icf_path=icf_path, worker_processes=0)
