import logging
import pathlib

import numpy as np

from bio2zarr import constants, core, vcz

logger = logging.getLogger(__name__)


class TskitFormat(vcz.Source):
    @core.requires_optional_dependency("tskit", "tskit")
    def __init__(
        self,
        ts_path,
        individuals_nodes=None,
        sample_ids=None,
        contig_id=None,
        isolated_as_missing=False,
    ):
        import tskit

        self._path = ts_path
        self.ts = tskit.load(ts_path)
        self.contig_id = contig_id if contig_id is not None else "1"
        self.isolated_as_missing = isolated_as_missing

        self.positions = self.ts.sites_position

        if individuals_nodes is None:
            individuals_nodes = self.ts.individuals_nodes

        self._num_samples = individuals_nodes.shape[0]
        if self._num_samples < 1:
            raise ValueError("individuals_nodes must have at least one sample")
        self.max_ploidy = individuals_nodes.shape[1]
        if sample_ids is None:
            sample_ids = [f"tsk_{j}" for j in range(self._num_samples)]
        elif len(sample_ids) != self._num_samples:
            raise ValueError(
                f"Length of sample_ids ({len(sample_ids)}) does not match "
                f"number of samples ({self._num_samples})"
            )

        self._samples = [vcz.Sample(id=sample_id) for sample_id in sample_ids]

        self.tskit_samples = np.unique(individuals_nodes[individuals_nodes >= 0])
        if len(self.tskit_samples) < 1:
            raise ValueError("individuals_nodes must have at least one valid sample")
        node_id_to_index = {node_id: i for i, node_id in enumerate(self.tskit_samples)}
        valid_mask = individuals_nodes >= 0
        self.sample_indices, self.ploidy_indices = np.where(valid_mask)
        self.genotype_indices = np.array(
            [node_id_to_index[node_id] for node_id in individuals_nodes[valid_mask]]
        )

    @property
    def path(self):
        return self._path

    @property
    def num_records(self):
        return self.ts.num_sites

    @property
    def num_samples(self):
        return self._num_samples

    @property
    def samples(self):
        return self._samples

    @property
    def root_attrs(self):
        return {}

    @property
    def contigs(self):
        return [vcz.Contig(id=self.contig_id)]

    def iter_contig(self, start, stop):
        yield from (0 for _ in range(start, stop))

    def iter_field(self, field_name, shape, start, stop):
        if field_name == "position":
            for pos in self.ts.sites_position[start:stop]:
                yield int(pos)
        else:
            raise ValueError(f"Unknown field {field_name}")

    def iter_alleles_and_genotypes(self, start, stop, shape, num_alleles):
        # All genotypes in tskit are considered phased
        phased = np.ones(shape[:-1], dtype=bool)

        for variant in self.ts.variants(
            isolated_as_missing=self.isolated_as_missing,
            left=self.positions[start],
            right=self.positions[stop] if stop < self.num_records else None,
            samples=self.tskit_samples,
        ):
            gt = np.full(shape, constants.INT_FILL, dtype=np.int8)
            alleles = np.full(num_alleles, constants.STR_FILL, dtype="O")
            for i, allele in enumerate(variant.alleles):
                # None is returned by tskit in the case of a missing allele
                if allele is None:
                    continue
                assert i < num_alleles
                alleles[i] = allele

            gt[self.sample_indices, self.ploidy_indices] = variant.genotypes[
                self.genotype_indices
            ]

            yield alleles, (gt, phased)

    def generate_schema(
        self,
        variants_chunk_size=None,
        samples_chunk_size=None,
    ):
        n = self.num_samples
        m = self.ts.num_sites

        # Determine max number of alleles
        max_alleles = 0
        for site in self.ts.sites():
            states = {site.ancestral_state}
            for mut in site.mutations:
                states.add(mut.derived_state)
            max_alleles = max(len(states), max_alleles)

        logging.info(f"Scanned tskit with {n} samples and {m} variants")
        logging.info(
            f"Maximum ploidy: {self.max_ploidy}, maximum alleles: {max_alleles}"
        )

        dimensions = {
            "variants": vcz.VcfZarrDimension(
                size=m, chunk_size=variants_chunk_size or vcz.DEFAULT_VARIANT_CHUNK_SIZE
            ),
            "samples": vcz.VcfZarrDimension(
                size=n, chunk_size=samples_chunk_size or vcz.DEFAULT_SAMPLE_CHUNK_SIZE
            ),
            "ploidy": vcz.VcfZarrDimension(size=self.max_ploidy),
            "alleles": vcz.VcfZarrDimension(size=max_alleles),
        }

        schema_instance = vcz.VcfZarrSchema(
            format_version=vcz.ZARR_SCHEMA_FORMAT_VERSION,
            dimensions=dimensions,
            fields=[],
        )

        logger.info(
            "Generating schema with chunks="
            f"{schema_instance.dimensions['variants'].chunk_size}, "
            f"{schema_instance.dimensions['samples'].chunk_size}"
        )

        # Check if positions will fit in i4 (max ~2.1 billion)
        min_position = 0
        max_position = 0
        if self.ts.num_sites > 0:
            min_position = np.min(self.ts.sites_position)
            max_position = np.max(self.ts.sites_position)

        array_specs = [
            vcz.ZarrArraySpec(
                source="position",
                name="variant_position",
                dtype=core.min_int_dtype(min_position, max_position),
                dimensions=["variants"],
                description="Position of each variant",
            ),
            vcz.ZarrArraySpec(
                source=None,
                name="variant_allele",
                dtype="O",
                dimensions=["variants", "alleles"],
                description="Alleles for each variant",
            ),
            vcz.ZarrArraySpec(
                source=None,
                name="variant_contig",
                dtype=core.min_int_dtype(0, len(self.contigs)),
                dimensions=["variants"],
                description="Contig/chromosome index for each variant",
            ),
            vcz.ZarrArraySpec(
                source=None,
                name="call_genotype_phased",
                dtype="bool",
                dimensions=["variants", "samples"],
                description="Whether the genotype is phased",
                compressor=vcz.DEFAULT_ZARR_COMPRESSOR_BOOL.get_config(),
            ),
            vcz.ZarrArraySpec(
                source=None,
                name="call_genotype",
                dtype=core.min_int_dtype(constants.INT_FILL, max_alleles - 1),
                dimensions=["variants", "samples", "ploidy"],
                description="Genotype for each variant and sample",
                compressor=vcz.DEFAULT_ZARR_COMPRESSOR_GENOTYPES.get_config(),
            ),
            vcz.ZarrArraySpec(
                source=None,
                name="call_genotype_mask",
                dtype="bool",
                dimensions=["variants", "samples", "ploidy"],
                description="Mask for each genotype call",
                compressor=vcz.DEFAULT_ZARR_COMPRESSOR_BOOL.get_config(),
            ),
        ]
        schema_instance.fields = array_specs
        return schema_instance


def convert(
    ts_path,
    zarr_path,
    *,
    individuals_nodes=None,
    sample_ids=None,
    contig_id=None,
    isolated_as_missing=False,
    variants_chunk_size=None,
    samples_chunk_size=None,
    worker_processes=1,
    show_progress=False,
):
    tskit_format = TskitFormat(
        ts_path,
        individuals_nodes=individuals_nodes,
        sample_ids=sample_ids,
        contig_id=contig_id,
        isolated_as_missing=isolated_as_missing,
    )
    schema_instance = tskit_format.generate_schema(
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
    )
    zarr_path = pathlib.Path(zarr_path)
    vzw = vcz.VcfZarrWriter(TskitFormat, zarr_path)
    # Rough heuristic to split work up enough to keep utilisation high
    target_num_partitions = max(1, worker_processes * 4)
    vzw.init(
        tskit_format,
        target_num_partitions=target_num_partitions,
        schema=schema_instance,
    )
    vzw.encode_all_partitions(
        worker_processes=worker_processes,
        show_progress=show_progress,
    )
    vzw.finalise(show_progress)
    vzw.create_index()
