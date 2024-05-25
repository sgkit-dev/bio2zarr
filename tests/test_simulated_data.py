import sys

import numpy.testing as nt
import pysam
import pytest
import sgkit as sg

from bio2zarr import vcf2zarr


@pytest.mark.skipif(sys.platform == "darwin", reason="msprime OSX pip packages broken")
class TestTskitRoundTripVcf:
    @pytest.mark.parametrize("ploidy", [1, 2, 3, 4])
    def test_ploidy(self, ploidy, tmp_path):
        # FIXME importing here so pytest.skip avoids importing msprime.
        import msprime

        ts = msprime.sim_ancestry(
            2,
            population_size=10**4,
            ploidy=ploidy,
            sequence_length=100_000,
            random_seed=42,
        )
        tables = ts.dump_tables()
        for u in ts.samples():
            site = tables.sites.add_row(u + 1, "A")
            tables.mutations.add_row(site, derived_state="T", node=u)
        ts = tables.tree_sequence()
        vcf_file = tmp_path / "sim.vcf"
        with open(vcf_file, "w") as f:
            ts.write_vcf(f)
        # This also compresses the input file
        pysam.tabix_index(str(vcf_file), preset="vcf")
        out = tmp_path / "example.vcf.zarr"
        vcf2zarr.convert([tmp_path / "sim.vcf.gz"], out)
        ds = sg.load_dataset(out)
        assert ds.sizes["ploidy"] == ploidy
        assert ds.sizes["variants"] == ts.num_sites
        assert ds.sizes["samples"] == ts.num_individuals
        # Msprime guarantees that this will be true.
        nt.assert_array_equal(
            ts.genotype_matrix().reshape((ts.num_sites, ts.num_individuals, ploidy)),
            ds.call_genotype.values,
        )
        nt.assert_equal(ds.variant_allele[:, 0].values, "A")
        nt.assert_equal(ds.variant_allele[:, 1].values, "T")
        nt.assert_equal(ds.variant_position, ts.sites_position)
