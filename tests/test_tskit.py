from unittest import mock

import msprime
import numpy as np
import numpy.testing as nt
import pytest
import sgkit as sg
import tskit
import xarray.testing as xt
import zarr

from bio2zarr import tskit as tsk
from bio2zarr import vcf
from bio2zarr.zarr_utils import STRING_DTYPE_NAME


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


def tskit_model_mapping(ind_nodes, ind_names=None):
    if ind_names is None:
        ind_names = ["tsk{j}" for j in range(len(ind_nodes))]
    return tskit.VcfModelMapping(ind_nodes, ind_names)


def add_mutations(ts):
    # Add some mutation to the tree sequence. This guarantees that
    # we have variation at all sites > 0.
    tables = ts.dump_tables()
    samples = ts.samples()
    states = "ACGT"
    for j in range(1, int(ts.sequence_length) - 1):
        site = tables.sites.add_row(j, ancestral_state=states[j % 4])
        tables.mutations.add_row(
            site=site,
            derived_state=states[(j + 1) % 4],
            node=samples[j % ts.num_samples],
        )
    return tables.tree_sequence()


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


def insert_branch_sites(ts, m=1):
    if m == 0:
        return ts
    tables = ts.dump_tables()
    tables.sites.clear()
    tables.mutations.clear()
    for tree in ts.trees():
        left, right = tree.interval
        delta = (right - left) / (m * len(list(tree.nodes())))
        x = left
        for u in tree.nodes():
            if tree.parent(u) != tskit.NULL:
                for _ in range(m):
                    site = tables.sites.add_row(position=x, ancestral_state="0")
                    tables.mutations.add_row(site=site, node=u, derived_state="1")
                    x += delta
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
        nt.assert_array_equal(pos, [10, 20, 30])

    def test_alleles(self, conversion):
        ts, zroot = conversion
        alleles = zroot["variant_allele"][:]
        assert alleles.shape == (3, 2)
        assert alleles.dtype.kind == STRING_DTYPE_NAME
        nt.assert_array_equal(alleles, [["A", "TTTT"], ["CCC", "G"], ["G", "AA"]])

    def test_variant_length(self, conversion):
        ts, zroot = conversion
        lengths = zroot["variant_length"][:]
        assert lengths.shape == (3,)
        assert lengths.dtype == np.int8
        nt.assert_array_equal(lengths, [1, 3, 1])

    def test_genotypes(self, conversion):
        ts, zroot = conversion
        genotypes = zroot["call_genotype"][:]
        assert genotypes.shape == (3, 4, 1)
        assert genotypes.dtype == np.int8
        nt.assert_array_equal(
            genotypes,
            [[[1], [1], [0], [0]], [[0], [0], [1], [1]], [[1], [0], [0], [0]]],
        )

    def test_phased(self, conversion):
        ts, zroot = conversion
        phased = zroot["call_genotype_phased"][:]
        assert phased.shape == (3, 4)
        assert phased.dtype == "bool"
        assert np.all(phased)

    def test_contig_id(self, conversion):
        ts, zroot = conversion
        contigs = zroot["contig_id"][:]
        assert contigs.shape == (1,)
        assert contigs.dtype.kind == STRING_DTYPE_NAME
        nt.assert_array_equal(contigs, ["1"])

    def test_variant_contig(self, conversion):
        ts, zroot = conversion
        contig = zroot["variant_contig"][:]
        assert contig.shape == (3,)
        assert contig.dtype == np.int8
        nt.assert_array_equal(contig, [0, 0, 0])

    def test_sample_id(self, conversion):
        ts, zroot = conversion
        samples = zroot["sample_id"][:]
        assert samples.shape == (4,)
        assert samples.dtype.kind == STRING_DTYPE_NAME
        nt.assert_array_equal(samples, ["tsk_0", "tsk_1", "tsk_2", "tsk_3"])

    def test_region_index(self, conversion):
        ts, zroot = conversion
        region_index = zroot["region_index"][:]
        assert region_index.shape == (1, 6)
        assert region_index.dtype == np.int8
        nt.assert_array_equal(region_index, [[0, 0, 10, 30, 30, 3]])

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
    def fx_simple_ts(self):
        return simple_ts(add_individuals=True)

    @pytest.fixture()
    def fx_ts_2_diploids(self):
        ts = msprime.sim_ancestry(2, sequence_length=10, random_seed=42)
        return add_mutations(ts)

    @pytest.fixture()
    def fx_ts_isolated_samples(self):
        tables = tskit.Tree.generate_balanced(2, span=10).tree_sequence.dump_tables()
        # This also tests sample nodes that are not a single block at
        # the start of the nodes table.
        tables.nodes.add_row(time=0, flags=tskit.NODE_IS_SAMPLE)
        return insert_branch_sites(tables.tree_sequence())

    def test_path_or_ts_input(self, tmp_path, fx_simple_ts):
        f1 = tsk.TskitFormat(fx_simple_ts)
        ts_path = tmp_path / "trees.ts"
        fx_simple_ts.dump(ts_path)
        f2 = tsk.TskitFormat(ts_path)
        f1.ts.tables.assert_equals(f2.ts.tables)

    def test_small_position_dtype(self):
        tables = tskit.TableCollection(sequence_length=100)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.sites.add_row(position=10, ancestral_state="A")
        tables.sites.add_row(position=20, ancestral_state="C")
        ts = tables.tree_sequence()
        format_obj_small = tsk.TskitFormat(ts)
        schema_small = format_obj_small.generate_schema()

        position_field = next(
            f for f in schema_small.fields if f.name == "variant_position"
        )
        assert position_field.dtype == "i1"

    def test_large_position_dtype(self):
        tables = tskit.TableCollection(sequence_length=3_000_000_000)
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        tables.sites.add_row(position=10, ancestral_state="A")
        tables.sites.add_row(position=np.iinfo(np.int32).max + 1, ancestral_state="C")
        ts = tables.tree_sequence()

        format_obj_large = tsk.TskitFormat(ts)
        schema_large = format_obj_large.generate_schema()

        position_field = next(
            f for f in schema_large.fields if f.name == "variant_position"
        )
        assert position_field.dtype == "i8"

    def test_initialization_defaults(self, fx_simple_ts):
        format_obj = tsk.TskitFormat(fx_simple_ts)
        assert format_obj.path is None
        assert format_obj.ts.num_sites == fx_simple_ts.num_sites
        assert format_obj.contig_id == "1"
        assert not format_obj.isolated_as_missing

    def test_initialization_params(self, fx_simple_ts):
        format_obj = tsk.TskitFormat(
            fx_simple_ts,
            contig_id="chr1",
            isolated_as_missing=True,
        )
        assert format_obj.contig_id == "chr1"
        assert format_obj.isolated_as_missing

    def test_basic_properties(self, fx_ts_2_diploids):
        format_obj = tsk.TskitFormat(fx_ts_2_diploids)

        assert format_obj.num_records == format_obj.ts.num_sites
        assert format_obj.num_samples == 2  # Two individuals
        assert len(format_obj.samples) == 2
        assert format_obj.samples[0].id == "tsk_0"
        assert format_obj.samples[1].id == "tsk_1"

        assert format_obj.root_attrs == {}

        contigs = format_obj.contigs
        assert len(contigs) == 1
        assert contigs[0].id == "1"

    def test_custom_sample_ids(self, fx_ts_2_diploids):
        custom_ids = ["sW", "sX"]
        model_mapping = fx_ts_2_diploids.map_to_vcf_model(individual_names=custom_ids)
        format_obj = tsk.TskitFormat(fx_ts_2_diploids, model_mapping=model_mapping)

        assert format_obj.num_samples == 2
        assert len(format_obj.samples) == 2
        assert format_obj.samples[0].id == "sW"
        assert format_obj.samples[1].id == "sX"

    def test_schema_generation(self, fx_simple_ts):
        format_obj = tsk.TskitFormat(fx_simple_ts)

        schema = format_obj.generate_schema()
        assert schema.dimensions["variants"].size == 3
        assert schema.dimensions["samples"].size == 4
        assert schema.dimensions["ploidy"].size == 1
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

    def test_iter_contig(self, fx_simple_ts):
        format_obj = tsk.TskitFormat(fx_simple_ts)
        contig_indices = list(format_obj.iter_contig(1, 3))
        assert contig_indices == [0, 0]

    def test_iter_field(self, fx_simple_ts):
        format_obj = tsk.TskitFormat(fx_simple_ts)
        positions = list(format_obj.iter_field("position", None, 0, 3))
        assert positions == [10, 20, 30]
        positions = list(format_obj.iter_field("position", None, 1, 3))
        assert positions == [20, 30]
        with pytest.raises(ValueError, match="Unknown field"):
            list(format_obj.iter_field("unknown_field", None, 0, 3))

    def test_zero_samples(self, fx_simple_ts):
        model_mapping = tskit_model_mapping(np.array([]))
        with pytest.raises(ValueError, match="at least one sample"):
            tsk.TskitFormat(fx_simple_ts, model_mapping=model_mapping)

    def test_no_valid_samples(self, fx_simple_ts):
        model_mapping = fx_simple_ts.map_to_vcf_model()
        model_mapping.individuals_nodes[:] = -1
        with pytest.raises(ValueError, match="at least one valid sample"):
            tsk.TskitFormat(fx_simple_ts, model_mapping=model_mapping)

    def test_model_size_mismatch(self, fx_simple_ts):
        model_mapping = fx_simple_ts.map_to_vcf_model()
        model_mapping.individuals_name = ["x"]
        with pytest.raises(ValueError, match="match number of samples"):
            tsk.TskitFormat(fx_simple_ts, model_mapping=model_mapping)

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
    def test_iter_alleles_and_genotypes(self, fx_simple_ts, ind_nodes, expected_gts):
        model_mapping = tskit_model_mapping(ind_nodes)
        format_obj = tsk.TskitFormat(fx_simple_ts, model_mapping=model_mapping)

        shape = (2, 2)  # (num_samples, max_ploidy)
        results = list(format_obj.iter_alleles_and_genotypes(0, 3, shape, 2))

        assert len(results) == 3

        for i, variant_data in enumerate(results):
            if i == 0:
                assert variant_data.variant_length == 1
                nt.assert_array_equal(variant_data.alleles, ("A", "TTTT"))
            elif i == 1:
                assert variant_data.variant_length == 3
                nt.assert_array_equal(variant_data.alleles, ("CCC", "G"))
            elif i == 2:
                assert variant_data.variant_length == 1
                nt.assert_array_equal(variant_data.alleles, ("G", "AA"))

            nt.assert_array_equal(variant_data.genotypes, expected_gts[i])
            assert np.all(variant_data.phased)

    def test_iter_alleles_and_genotypes_missing_node(self, fx_ts_2_diploids):
        # Test with node ID that doesn't exist in tree sequence (out of range)
        ind_nodes = np.array([[10, 11], [12, 13]], dtype=np.int32)
        model_mapping = tskit_model_mapping(ind_nodes)
        format_obj = tsk.TskitFormat(fx_ts_2_diploids, model_mapping=model_mapping)
        shape = (2, 2)
        with pytest.raises(
            tskit.LibraryError, match="out of bounds"
        ):  # Node ID 10 doesn't exist
            list(format_obj.iter_alleles_and_genotypes(0, 1, shape, 2))

    def test_isolated_as_missing(self, fx_ts_isolated_samples):
        ind_nodes = np.array([[0], [1], [3]])
        model_mapping = tskit_model_mapping(ind_nodes)

        format_obj_default = tsk.TskitFormat(
            fx_ts_isolated_samples,
            model_mapping=model_mapping,
            isolated_as_missing=False,
        )
        shape = (3, 1)  # (num_samples, max_ploidy)
        results_default = list(
            format_obj_default.iter_alleles_and_genotypes(0, 1, shape, 2)
        )

        assert len(results_default) == 1
        variant_data_default = results_default[0]
        nt.assert_array_equal(variant_data_default.alleles, ("0", "1"))

        # Sample 2 should have the ancestral state (0) when isolated_as_missing=False
        expected_gt_default = np.array([[1], [0], [0]])
        nt.assert_array_equal(variant_data_default.genotypes, expected_gt_default)

        format_obj_missing = tsk.TskitFormat(
            fx_ts_isolated_samples,
            model_mapping=model_mapping,
            isolated_as_missing=True,
        )
        results_missing = list(
            format_obj_missing.iter_alleles_and_genotypes(0, 1, shape, 2)
        )

        assert len(results_missing) == 1
        variant_data_missing = results_missing[0]
        assert variant_data_missing.variant_length == 1
        nt.assert_array_equal(variant_data_missing.alleles, ("0", "1"))

        # Individual 2 should have missing values (-1) when isolated_as_missing=True
        expected_gt_missing = np.array([[1], [0], [-1]])
        nt.assert_array_equal(variant_data_missing.genotypes, expected_gt_missing)

    def test_genotype_dtype_i1(self):
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

        format_obj = tsk.TskitFormat(tree_sequence)
        schema = format_obj.generate_schema()
        call_genotype_spec = next(s for s in schema.fields if s.name == "call_genotype")
        assert call_genotype_spec.dtype == "i1"

    def test_genotype_dtype_i4(self):
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

        format_obj = tsk.TskitFormat(tree_sequence)
        schema = format_obj.generate_schema()
        call_genotype_spec = next(s for s in schema.fields if s.name == "call_genotype")
        assert call_genotype_spec.dtype == "i4"


@pytest.mark.parametrize(
    "ts",
    [
        # Standard individuals-with-a-given-ploidy situation
        add_mutations(
            msprime.sim_ancestry(4, ploidy=1, sequence_length=10, random_seed=42)
        ),
        add_mutations(
            msprime.sim_ancestry(2, ploidy=2, sequence_length=10, random_seed=42)
        ),
        add_mutations(
            msprime.sim_ancestry(3, ploidy=12, sequence_length=10, random_seed=142)
        ),
        # No individuals, ploidy1
        add_mutations(msprime.simulate(4, length=10, random_seed=412)),
    ],
)
def test_against_tskit_vcf_output(ts, tmp_path):
    vcf_path = tmp_path / "ts.vcf"
    with open(vcf_path, "w") as f:
        ts.write_vcf(f)

    tskit_zarr = tmp_path / "tskit.zarr"
    vcf_zarr = tmp_path / "vcf.zarr"
    tsk.convert(ts, tskit_zarr, worker_processes=0)

    vcf.convert([vcf_path], vcf_zarr, worker_processes=0)
    ds1 = sg.load_dataset(tskit_zarr)
    ds2 = (
        sg.load_dataset(vcf_zarr)
        .drop_dims("filters")
        .drop_vars(
            ["variant_id", "variant_id_mask", "variant_quality", "contig_length"]
        )
    )
    xt.assert_equal(ds1, ds2)


def assert_ts_ds_equal(ts, ds, ploidy=2):
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
    # Specialised for the limited form of mutations used here
    nt.assert_equal(
        ds.variant_allele[:, 0].values, [site.ancestral_state for site in ts.sites()]
    )
    nt.assert_equal(
        ds.variant_allele[:, 1].values,
        [mutation.derived_state for mutation in ts.mutations()],
    )
    nt.assert_equal(ds.variant_position, ts.sites_position)


@pytest.mark.parametrize("worker_processes", [0, 1, 2, 15])
def test_workers(tmp_path, worker_processes):
    ts = msprime.sim_ancestry(10, sequence_length=1000, random_seed=42)
    ts = add_mutations(ts)
    out = tmp_path / "tskit.zarr"
    tsk.convert(ts, out, worker_processes=worker_processes)
    ds = sg.load_dataset(out)
    assert_ts_ds_equal(ts, ds)
