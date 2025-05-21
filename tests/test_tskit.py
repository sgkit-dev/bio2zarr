from unittest import mock

import numpy as np
import pytest
import sgkit as sg
import tskit
import xarray.testing as xt
import zarr

from bio2zarr import tskit as tsk
from bio2zarr import vcf


def test_missing_dependency():
    with mock.patch(
        "importlib.import_module",
        side_effect=ImportError("No module named 'tskit'"),
    ):
        with pytest.raises(ImportError) as exc_info:
            tsk.convert(
                "UNUSED_PATH",
                "UNUSED_PATH",
            )
        assert (
            "This process requires the optional tskit module. Install "
            "it with: pip install bio2zarr[tskit]" in str(exc_info.value)
        )


def simple_ts(add_individuals=False):
    tables = tskit.TableCollection(sequence_length=100)
    for _ in range(4):
        ind = -1
        if add_individuals:
            ind = tables.individuals.add_row()
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0, individual=ind)
    tables.nodes.add_row(flags=0, time=1)  # MRCA for 0,1
    tables.nodes.add_row(flags=0, time=1)  # MRCA for 2,3
    tables.edges.add_row(left=0, right=100, parent=4, child=0)
    tables.edges.add_row(left=0, right=100, parent=4, child=1)
    tables.edges.add_row(left=0, right=100, parent=5, child=2)
    tables.edges.add_row(left=0, right=100, parent=5, child=3)
    site_id = tables.sites.add_row(position=10, ancestral_state="A")
    tables.mutations.add_row(site=site_id, node=4, derived_state="TTTT")
    site_id = tables.sites.add_row(position=20, ancestral_state="CCC")
    tables.mutations.add_row(site=site_id, node=5, derived_state="G")
    site_id = tables.sites.add_row(position=30, ancestral_state="G")
    tables.mutations.add_row(site=site_id, node=0, derived_state="AA")

    tables.sort()
    return tables.tree_sequence()


class TestSimpleTs:
    @pytest.fixture()
    def conversion(self, tmp_path):
        ts = simple_ts()
        zarr_path = tmp_path / "test_output.vcz"
        tsk.convert(ts, zarr_path)
        zroot = zarr.open(zarr_path, mode="r")
        return ts, zroot

    def test_position(self, conversion):
        ts, zroot = conversion

        pos = zroot["variant_position"][:]
        assert pos.shape == (3,)
        assert pos.dtype == np.int8
        assert np.array_equal(pos, [10, 20, 30])

    def test_alleles(self, conversion):
        ts, zroot = conversion
        alleles = zroot["variant_allele"][:]
        assert alleles.shape == (3, 2)
        assert alleles.dtype == "O"
        assert np.array_equal(alleles, [["A", "TTTT"], ["CCC", "G"], ["G", "AA"]])

    def test_variant_length(self, conversion):
        ts, zroot = conversion
        lengths = zroot["variant_length"][:]
        assert lengths.shape == (3,)
        assert lengths.dtype == np.int8
        assert np.array_equal(lengths, [1, 3, 1])

    def test_genotypes(self, conversion):
        ts, zroot = conversion
        genotypes = zroot["call_genotype"][:]
        assert genotypes.shape == (3, 4, 1)
        assert genotypes.dtype == np.int8
        assert np.array_equal(
            genotypes,
            [[[1], [1], [0], [0]], [[0], [0], [1], [1]], [[1], [0], [0], [0]]],
        )

    def test_phased(self, conversion):
        ts, zroot = conversion
        phased = zroot["call_genotype_phased"][:]
        assert phased.shape == (3, 4)
        assert phased.dtype == "bool"
        assert np.all(~phased)

    def test_contig_id(self, conversion):
        ts, zroot = conversion
        contigs = zroot["contig_id"][:]
        assert contigs.shape == (1,)
        assert contigs.dtype == "O"
        assert np.array_equal(contigs, ["1"])

    def test_variant_contig(self, conversion):
        ts, zroot = conversion
        contig = zroot["variant_contig"][:]
        assert contig.shape == (3,)
        assert contig.dtype == np.int8
        assert np.array_equal(contig, [0, 0, 0])

    def test_sample_id(self, conversion):
        ts, zroot = conversion
        samples = zroot["sample_id"][:]
        assert samples.shape == (4,)
        assert samples.dtype == "O"
        assert np.array_equal(samples, ["tsk_0", "tsk_1", "tsk_2", "tsk_3"])

    def test_region_index(self, conversion):
        ts, zroot = conversion
        region_index = zroot["region_index"][:]
        assert region_index.shape == (1, 6)
        assert region_index.dtype == np.int8
        assert np.array_equal(region_index, [[0, 0, 10, 30, 30, 3]])

    def test_fields(self, conversion):
        ts, zroot = conversion
        assert set(zroot.array_keys()) == {
            "variant_position",
            "variant_allele",
            "variant_length",
            "call_genotype",
            "call_genotype_phased",
            "call_genotype_mask",
            "contig_id",
            "variant_contig",
            "sample_id",
            "region_index",
        }


class TestTskitFormat:
    """Unit tests for TskitFormat without using full conversion."""

    @pytest.fixture()
    def simple_ts(self, tmp_path):
        tables = tskit.TableCollection(sequence_length=100)
        tables.individuals.add_row()
        tables.individuals.add_row()
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
        tables.mutations.add_row(site=site_id, node=4, derived_state="TT")
        site_id = tables.sites.add_row(position=20, ancestral_state="CCC")
        tables.mutations.add_row(site=site_id, node=5, derived_state="G")
        site_id = tables.sites.add_row(position=30, ancestral_state="G")
        tables.mutations.add_row(site=site_id, node=0, derived_state="A")
        tables.sort()
        tree_sequence = tables.tree_sequence()
        ts_path = tmp_path / "test.trees"
        tree_sequence.dump(ts_path)
        return ts_path, tree_sequence

    @pytest.fixture()
    def no_individuals_ts(self, tmp_path):
        tables = tskit.TableCollection(sequence_length=100)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
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
        tables.sort()
        tree_sequence = tables.tree_sequence()
        ts_path = tmp_path / "no_individuals.trees"
        tree_sequence.dump(ts_path)
        return ts_path, tree_sequence

    def test_position_dtype_selection(self, tmp_path):
        tables = tskit.TableCollection(sequence_length=100)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.sites.add_row(position=10, ancestral_state="A")
        tables.sites.add_row(position=20, ancestral_state="C")
        ts_small = tables.tree_sequence()
        ts_path_small = tmp_path / "small_positions.trees"
        ts_small.dump(ts_path_small)

        tables = tskit.TableCollection(sequence_length=3_000_000_000)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.sites.add_row(position=10, ancestral_state="A")
        tables.sites.add_row(position=np.iinfo(np.int32).max + 1, ancestral_state="C")
        ts_large = tables.tree_sequence()
        ts_path_large = tmp_path / "large_positions.trees"
        ts_large.dump(ts_path_large)

        ind_nodes = np.array([[0], [1]])
        format_obj_small = tsk.TskitFormat(ts_path_small, individuals_nodes=ind_nodes)
        schema_small = format_obj_small.generate_schema()

        position_field = next(
            f for f in schema_small.fields if f.name == "variant_position"
        )
        assert position_field.dtype == "i1"

        format_obj_large = tsk.TskitFormat(ts_path_large, individuals_nodes=ind_nodes)
        schema_large = format_obj_large.generate_schema()

        position_field = next(
            f for f in schema_large.fields if f.name == "variant_position"
        )
        assert position_field.dtype == "i8"

    def test_initialization(self, simple_ts):
        ts_path, tree_sequence = simple_ts

        # Test with default parameters
        format_obj = tsk.TskitFormat(ts_path)
        assert format_obj.path == ts_path
        assert format_obj.ts.num_sites == tree_sequence.num_sites
        assert format_obj.contig_id == "1"
        assert not format_obj.isolated_as_missing

        # Test with custom parameters
        format_obj = tsk.TskitFormat(
            ts_path,
            sample_ids=["ind1", "ind2"],
            contig_id="chr1",
            isolated_as_missing=True,
        )
        assert format_obj.contig_id == "chr1"
        assert format_obj.isolated_as_missing
        assert format_obj.path == ts_path
        assert format_obj.samples[0].id == "ind1"
        assert format_obj.samples[1].id == "ind2"

    def test_basic_properties(self, simple_ts):
        ts_path, _ = simple_ts
        format_obj = tsk.TskitFormat(ts_path)

        assert format_obj.num_records == format_obj.ts.num_sites
        assert format_obj.num_samples == 2  # Two individuals
        assert len(format_obj.samples) == 2
        assert format_obj.samples[0].id == "tsk_0"
        assert format_obj.samples[1].id == "tsk_1"

        assert format_obj.root_attrs == {}

        contigs = format_obj.contigs
        assert len(contigs) == 1
        assert contigs[0].id == "1"

    def test_custom_sample_ids(self, simple_ts):
        ts_path, _ = simple_ts
        custom_ids = ["sample_X", "sample_Y"]
        format_obj = tsk.TskitFormat(ts_path, sample_ids=custom_ids)

        assert format_obj.num_samples == 2
        assert len(format_obj.samples) == 2
        assert format_obj.samples[0].id == "sample_X"
        assert format_obj.samples[1].id == "sample_Y"

    def test_sample_id_length_mismatch(self, simple_ts):
        ts_path, _ = simple_ts
        # Wrong number of sample IDs
        with pytest.raises(ValueError, match="Length of sample_ids.*does not match"):
            tsk.TskitFormat(ts_path, sample_ids=["only_one_id"])

    def test_schema_generation(self, simple_ts):
        ts_path, _ = simple_ts
        format_obj = tsk.TskitFormat(ts_path)

        schema = format_obj.generate_schema()
        assert schema.dimensions["variants"].size == 3
        assert schema.dimensions["samples"].size == 2
        assert schema.dimensions["ploidy"].size == 2
        assert schema.dimensions["alleles"].size == 2  # A/T, C/G, G/A -> max is 2
        field_names = [field.name for field in schema.fields]
        assert "variant_position" in field_names
        assert "variant_allele" in field_names
        assert "variant_length" in field_names
        assert "variant_contig" in field_names
        assert "call_genotype" in field_names
        assert "call_genotype_phased" in field_names
        assert "call_genotype_mask" in field_names
        schema = format_obj.generate_schema(
            variants_chunk_size=10, samples_chunk_size=5
        )
        assert schema.dimensions["variants"].chunk_size == 10
        assert schema.dimensions["samples"].chunk_size == 5

    def test_iter_contig(self, simple_ts):
        ts_path, _ = simple_ts
        format_obj = tsk.TskitFormat(ts_path)
        contig_indices = list(format_obj.iter_contig(1, 3))
        assert contig_indices == [0, 0]

    def test_iter_field(self, simple_ts):
        ts_path, _ = simple_ts
        format_obj = tsk.TskitFormat(ts_path)
        positions = list(format_obj.iter_field("position", None, 0, 3))
        assert positions == [10, 20, 30]
        positions = list(format_obj.iter_field("position", None, 1, 3))
        assert positions == [20, 30]
        with pytest.raises(ValueError, match="Unknown field"):
            list(format_obj.iter_field("unknown_field", None, 0, 3))

    @pytest.mark.parametrize(
        ("ind_nodes", "expected_gts"),
        [
            # Standard case: diploid samples with sequential node IDs
            (
                np.array([[0, 1], [2, 3]]),
                [[[1, 1], [0, 0]], [[0, 0], [1, 1]], [[1, 0], [0, 0]]],
            ),
            # Mixed ploidy: first sample diploid, second haploid
            (
                np.array([[0, 1], [2, -1]]),
                [[[1, 1], [0, -2]], [[0, 0], [1, -2]], [[1, 0], [0, -2]]],
            ),
            # Reversed order: nodes are not in sequential order
            (
                np.array([[2, 3], [0, 1]]),
                [[[0, 0], [1, 1]], [[1, 1], [0, 0]], [[0, 0], [1, 0]]],
            ),
            # Duplicate nodes: same node used multiple times
            (
                np.array([[0, 0], [2, 2]]),
                [[[1, 1], [0, 0]], [[0, 0], [1, 1]], [[1, 1], [0, 0]]],
            ),
            # Non-sample node: using node 4 which is an internal node (MRCA for 0,1)
            (
                np.array([[0, 4], [2, 3]]),
                [[[1, 1], [0, 0]], [[0, 0], [1, 1]], [[1, 0], [0, 0]]],
            ),
            # One individual with zero ploidy
            (
                np.array([[0, 1], [-1, -1]]),
                [[[1, 1], [-2, -2]], [[0, 0], [-2, -2]], [[1, 0], [-2, -2]]],
            ),
        ],
    )
    def test_iter_alleles_and_genotypes(self, simple_ts, ind_nodes, expected_gts):
        ts_path, _ = simple_ts

        format_obj = tsk.TskitFormat(ts_path, individuals_nodes=ind_nodes)

        shape = (2, 2)  # (num_samples, max_ploidy)
        results = list(format_obj.iter_alleles_and_genotypes(0, 3, shape, 2))

        assert len(results) == 3

        for i, variant_data in enumerate(results):
            if i == 0:
                assert variant_data.variant_length == 1
                assert np.array_equal(variant_data.alleles, ("A", "TT"))
            elif i == 1:
                assert variant_data.variant_length == 3
                assert np.array_equal(variant_data.alleles, ("CCC", "G"))
            elif i == 2:
                assert variant_data.variant_length == 1
                assert np.array_equal(variant_data.alleles, ("G", "A"))

            assert np.array_equal(
                variant_data.genotypes, expected_gts[i]
            ), f"Mismatch at variant {i}, expected {expected_gts[i]}, "
            f"got {variant_data.genotypes}"
            assert np.all(variant_data.phased)

    def test_iter_alleles_and_genotypes_errors(self, simple_ts):
        """Test error cases for iter_alleles_and_genotypes with invalid inputs."""
        ts_path, _ = simple_ts

        # Test with node ID that doesn't exist in tree sequence (out of range)
        invalid_nodes = np.array([[10, 11], [12, 13]], dtype=np.int32)
        format_obj = tsk.TskitFormat(ts_path, individuals_nodes=invalid_nodes)
        shape = (2, 2)
        with pytest.raises(
            tskit.LibraryError, match="out of bounds"
        ):  # Node ID 10 doesn't exist
            list(format_obj.iter_alleles_and_genotypes(0, 1, shape, 2))

        # Test with empty ind_nodes array (no samples)
        empty_nodes = np.zeros((0, 2), dtype=np.int32)
        with pytest.raises(
            ValueError, match="individuals_nodes must have at least one sample"
        ):
            format_obj = tsk.TskitFormat(ts_path, individuals_nodes=empty_nodes)

        # Test with all invalid nodes (-1)
        all_invalid = np.full((2, 2), -1, dtype=np.int32)
        with pytest.raises(
            ValueError, match="individuals_nodes must have at least one valid sample"
        ):
            format_obj = tsk.TskitFormat(ts_path, individuals_nodes=all_invalid)

    def test_isolated_as_missing(self, tmp_path):
        def insert_branch_sites(tsk, m=1):
            if m == 0:
                return tsk
            tables = tsk.dump_tables()
            tables.sites.clear()
            tables.mutations.clear()
            for tree in tsk.trees():
                left, right = tree.interval
                delta = (right - left) / (m * len(list(tree.nodes())))
                x = left
                for u in tree.nodes():
                    if tree.parent(u) != tskit.NULL:
                        for _ in range(m):
                            site = tables.sites.add_row(position=x, ancestral_state="0")
                            tables.mutations.add_row(
                                site=site, node=u, derived_state="1"
                            )
                            x += delta
            return tables.tree_sequence()

        tables = tskit.Tree.generate_balanced(2, span=10).tree_sequence.dump_tables()
        # This also tests sample nodes that are not a single block at
        # the start of the nodes table.
        tables.nodes.add_row(time=0, flags=tskit.NODE_IS_SAMPLE)
        tree_sequence = insert_branch_sites(tables.tree_sequence())

        ts_path = tmp_path / "isolated_sample.trees"
        tree_sequence.dump(ts_path)
        ind_nodes = np.array([[0], [1], [3]])
        format_obj_default = tsk.TskitFormat(
            ts_path, individuals_nodes=ind_nodes, isolated_as_missing=False
        )
        shape = (3, 1)  # (num_samples, max_ploidy)
        results_default = list(
            format_obj_default.iter_alleles_and_genotypes(0, 1, shape, 2)
        )

        assert len(results_default) == 1
        variant_data_default = results_default[0]
        assert np.array_equal(variant_data_default.alleles, ("0", "1"))

        # Sample 2 should have the ancestral state (0) when isolated_as_missing=False
        expected_gt_default = np.array([[1], [0], [0]])
        assert np.array_equal(variant_data_default.genotypes, expected_gt_default)

        format_obj_missing = tsk.TskitFormat(
            ts_path, individuals_nodes=ind_nodes, isolated_as_missing=True
        )
        results_missing = list(
            format_obj_missing.iter_alleles_and_genotypes(0, 1, shape, 2)
        )

        assert len(results_missing) == 1
        variant_data_missing = results_missing[0]
        assert variant_data_missing.variant_length == 1
        assert np.array_equal(variant_data_missing.alleles, ("0", "1"))

        # Individual 2 should have missing values (-1) when isolated_as_missing=True
        expected_gt_missing = np.array([[1], [0], [-1]])
        assert np.array_equal(variant_data_missing.genotypes, expected_gt_missing)

    def test_genotype_dtype_i1(self, tmp_path):
        tables = tskit.TableCollection(sequence_length=100)
        for _ in range(4):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        mrca = tables.nodes.add_row(flags=0, time=1)
        for i in range(4):
            tables.edges.add_row(left=0, right=100, parent=mrca, child=i)
        site_id = tables.sites.add_row(position=10, ancestral_state="A")
        tables.mutations.add_row(site=site_id, node=0, derived_state="T")
        tables.sort()
        tree_sequence = tables.tree_sequence()
        ts_path = tmp_path / "small_alleles.trees"
        tree_sequence.dump(ts_path)

        format_obj = tsk.TskitFormat(ts_path)
        schema = format_obj.generate_schema()
        call_genotype_spec = next(s for s in schema.fields if s.name == "call_genotype")
        assert call_genotype_spec.dtype == "i1"

    def test_genotype_dtype_i4(self, tmp_path):
        tables = tskit.TableCollection(sequence_length=100)
        for _ in range(4):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        mrca = tables.nodes.add_row(flags=0, time=1)
        for i in range(4):
            tables.edges.add_row(left=0, right=100, parent=mrca, child=i)
        site_id = tables.sites.add_row(position=10, ancestral_state="A")
        for i in range(32768):
            tables.mutations.add_row(site=site_id, node=0, derived_state=f"ALLELE_{i}")

        tables.sort()
        tree_sequence = tables.tree_sequence()
        ts_path = tmp_path / "large_alleles.trees"
        tree_sequence.dump(ts_path)

        format_obj = tsk.TskitFormat(ts_path)
        schema = format_obj.generate_schema()
        call_genotype_spec = next(s for s in schema.fields if s.name == "call_genotype")
        assert call_genotype_spec.dtype == "i4"


@pytest.mark.parametrize(
    "ts",
    [
        simple_ts(add_individuals=True),
        simple_ts(add_individuals=False),
    ],
)
def test_against_tskit_vcf_output(ts, tmp_path):
    vcf_path = tmp_path / "ts.vcf"
    ts_path = tmp_path / "ts.trees"
    ts.dump(ts_path)
    with open(vcf_path, "w") as f:
        ts.write_vcf(f)

    tskit_zarr = tmp_path / "tskit.zarr"
    vcf_zarr = tmp_path / "vcf.zarr"
    tsk.convert(ts_path, tskit_zarr)

    vcf.convert([vcf_path], vcf_zarr)
    ds1 = sg.load_dataset(tskit_zarr)
    ds2 = (
        sg.load_dataset(vcf_zarr)
        .drop_dims("filters")
        .drop_vars(
            ["variant_id", "variant_id_mask", "variant_quality", "contig_length"]
        )
    )
    xt.assert_equal(ds1, ds2)
