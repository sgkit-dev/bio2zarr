import msprime
import numpy as np
import numpy.testing as nt
import pysam
import pytest
import sgkit as sg

from bio2zarr import vcf as vcf_mod


def run_simulation(num_samples=2, ploidy=1, seed=42, sequence_length=100_000):
    ts = msprime.sim_ancestry(
        num_samples,
        population_size=10**4,
        ploidy=ploidy,
        sequence_length=sequence_length,
        random_seed=seed,
    )
    tables = ts.dump_tables()
    # Lazy hard coding of states here to make things simpler
    for u in range(ts.num_nodes - 1):
        site = tables.sites.add_row(u + 1, "A")
        tables.mutations.add_row(site, derived_state="T", node=u)
    return tables.tree_sequence()


def assert_ts_ds_equal(ts, ds, ploidy=1):
    assert ds.sizes["ploidy"] == ploidy
    assert ds.sizes["variants"] == ts.num_sites
    assert ds.sizes["samples"] == ts.num_individuals
    # Msprime guarantees that this will be true.
    nt.assert_array_equal(
        ts.genotype_matrix().reshape((ts.num_sites, ts.num_individuals, ploidy)),
        ds.call_genotype.values,
    )
    nt.assert_array_equal(
        ds.call_genotype_phased.values,
        np.ones((ts.num_sites, ts.num_individuals), dtype=bool),
    )
    nt.assert_equal(ds.variant_allele[:, 0].values, "A")
    nt.assert_equal(ds.variant_allele[:, 1].values, "T")
    nt.assert_equal(ds.variant_position, ts.sites_position)


def write_vcf(ts, vcf_path, contig_id="1", indexed=False):
    with open(vcf_path, "w") as f:
        ts.write_vcf(f, contig_id=contig_id)
    if indexed:
        # This also compresses the input file
        pysam.tabix_index(str(vcf_path), preset="vcf")
        vcf_path = vcf_path.with_suffix(vcf_path.suffix + ".gz")
    return vcf_path


class TestTskitRoundTripVcf:
    @pytest.mark.parametrize("ploidy", [1, 2, 3, 4])
    def test_ploidy(self, ploidy, tmp_path):
        ts = run_simulation(ploidy=ploidy)
        vcf_path = write_vcf(ts, tmp_path / "sim.vcf")
        out = tmp_path / "example.vcf.zarr"
        vcf_mod.convert([vcf_path], out)
        ds = sg.load_dataset(out)
        assert_ts_ds_equal(ts, ds, ploidy)

    @pytest.mark.parametrize(
        "contig_ids",
        [["ContigName"], ["1", "2"], ["2", "3", "1"], ["a", "b", "c", "d", "e"]],
    )
    def test_multi_contig(self, contig_ids, tmp_path):
        vcfs = []
        tss = {}
        for seed, contig_id in enumerate(contig_ids, 1):
            ts = run_simulation(num_samples=8, seed=seed)
            vcf_path = write_vcf(ts, tmp_path / f"{contig_id}.vcf", contig_id=contig_id)
            vcfs.append(vcf_path)
            tss[contig_id] = ts

        self.validate_tss_vcf_list(contig_ids, tss, vcfs, tmp_path)

    def validate_tss_vcf_list(self, contig_ids, tss, vcfs, tmp_path):
        out = tmp_path / "example.vcf.zarr"
        vcf_mod.convert(vcfs, out)
        ds = sg.load_dataset(out).set_index(
            variants=("variant_contig", "variant_position")
        )
        assert ds.sizes["ploidy"] == 1
        assert ds.sizes["contigs"] == len(contig_ids)
        # Files processed in order of sorted filename, and the contigs therefore
        # get sorted into this order.
        nt.assert_equal(ds["contig_id"].values, sorted(contig_ids))
        nt.assert_equal(
            ds["contig_length"].values, [ts.sequence_length for ts in tss.values()]
        )
        for contig_id in contig_ids:
            contig = list(ds.contig_id).index(contig_id)
            dss = ds.sel(variants=(contig, slice(0, None)))
            assert_ts_ds_equal(tss[contig_id], dss)

    @pytest.mark.parametrize("indexed", [True, False])
    def test_indexed(self, indexed, tmp_path):
        ts = run_simulation(num_samples=12, seed=34)
        vcf_path = write_vcf(ts, tmp_path / "sim.vcf", indexed=indexed)
        out = tmp_path / "example.vcf.zarr"
        vcf_mod.convert([vcf_path], out)
        ds = sg.load_dataset(out)
        assert_ts_ds_equal(ts, ds)

    @pytest.mark.parametrize("num_contigs", [2, 3, 6])
    def test_mixed_indexed(self, num_contigs, tmp_path):
        contig_ids = [f"x{j}" for j in range(num_contigs)]

        vcfs = []
        tss = {}
        for seed, contig_id in enumerate(contig_ids, 1):
            ts = run_simulation(num_samples=3, seed=seed)
            vcf_path = write_vcf(
                ts,
                tmp_path / f"{contig_id}.vcf",
                contig_id=contig_id,
                indexed=seed % 2 == 0,
            )
            vcfs.append(vcf_path)
            tss[contig_id] = ts

        self.validate_tss_vcf_list(contig_ids, tss, vcfs, tmp_path)


class TestIncompatibleContigs:
    def test_different_lengths(self, tmp_path):
        vcfs = []
        for length in [100_000, 99_999]:
            ts = run_simulation(sequence_length=length)
            vcf_path = write_vcf(ts, tmp_path / f"{length}.vcf", contig_id="1")
            vcfs.append(vcf_path)
        out = tmp_path / "example.vcf.zarr"
        with pytest.raises(ValueError, match="Incompatible contig definitions"):
            vcf_mod.convert(vcfs, out)
