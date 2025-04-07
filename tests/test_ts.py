import os
import tempfile

import numpy as np
import tskit
import zarr

from bio2zarr import ts


class TestTskit:
    def test_simple_tree_sequence(self, tmp_path):
        tables = tskit.TableCollection(sequence_length=100)
        tables.individuals.add_row(flags=0, location=(0, 0), metadata=b"")
        tables.individuals.add_row(flags=0, location=(0, 0), metadata=b"")
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=0)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=0)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=1)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=1)
        tables.nodes.add_row(flags=0, time=1)  # MRCA for 0,1
        tables.nodes.add_row(flags=0, time=1)  # MRCA for 2,3
        tables.edges.add_row(left=0, right=100, parent=4, child=0)
        tables.edges.add_row(left=0, right=100, parent=4, child=1)
        tables.edges.add_row(left=0, right=100, parent=5, child=2)
        tables.edges.add_row(left=0, right=100, parent=5, child=3)
        site_id = tables.sites.add_row(position=10, ancestral_state="A")
        tables.mutations.add_row(site=site_id, node=4, derived_state="T")
        site_id = tables.sites.add_row(position=20, ancestral_state="C")
        tables.mutations.add_row(site=site_id, node=5, derived_state="G")
        site_id = tables.sites.add_row(position=30, ancestral_state="G")
        tables.mutations.add_row(site=site_id, node=0, derived_state="A")
        tables.sort()
        tree_sequence = tables.tree_sequence()
        tree_sequence.dump(tmp_path / "test.trees")
        with tempfile.TemporaryDirectory() as tempdir:
            zarr_path = os.path.join(tempdir, "test_output.zarr")
            ts.convert(tmp_path / "test.trees", zarr_path, show_progress=False)
            zroot = zarr.open(zarr_path, mode="r")
            assert zroot["variant_position"].shape == (3,)
            assert list(zroot["variant_position"][:]) == [10, 20, 30]

            alleles = zroot["variant_allele"][:]
            assert np.array_equal(alleles, [["A", "T"], ["C", "G"], ["G", "A"]])

            genotypes = zroot["call_genotype"][:]
            assert np.array_equal(
                genotypes, [[[1, 1], [0, 0]], [[0, 0], [1, 1]], [[1, 0], [0, 0]]]
            )

            phased = zroot["call_genotype_phased"][:]
            assert np.all(phased)

            contigs = zroot["contig_id"][:]
            assert np.array_equal(contigs, ["1"])

            contig = zroot["variant_contig"][:]
            assert np.array_equal(contig, [0, 0, 0])

            samples = zroot["sample_id"][:]
            assert np.array_equal(samples, ["tsk_0", "tsk_1"])
