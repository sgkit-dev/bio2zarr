import logging
import pathlib

import numpy as np

from bio2zarr import constants, core, vcz

logger = logging.getLogger(__name__)


class TskitFormat(vcz.Source):
    @core.requires_optional_dependency("tskit", "tskit")
    def __init__(
        self,
        ts,
        *,
        model_mapping=None,
        contig_id=None,
        isolated_as_missing=False,
    ):
        import tskit

        self._path = None
        # Future versions here will need to deal with the complexities of
        # having lists of tree sequences for multiple chromosomes.
        if isinstance(ts, tskit.TreeSequence):
            self.ts = ts
        else:
            # input 'ts' is a path.
            self._path = ts
            logger.info(f"Loading from {ts}")
            self.ts = tskit.load(ts)
        logger.info(
            f"Input has {self.ts.num_individuals} individuals and "
            f"{self.ts.num_sites} sites"
        )

        self.contig_id = contig_id if contig_id is not None else "1"
        self.isolated_as_missing = isolated_as_missing

        self.positions = self.ts.sites_position

        if model_mapping is None:
            model_mapping = self.ts.map_to_vcf_model()

        individuals_nodes = model_mapping.individuals_nodes
        sample_ids = model_mapping.individuals_name

        self._num_samples = individuals_nodes.shape[0]
        logger.info(f"Converting for {self._num_samples} samples")
        if self._num_samples < 1:
            raise ValueError("individuals_nodes must have at least one sample")
        self.max_ploidy = individuals_nodes.shape[1]
        if len(sample_ids) != self._num_samples:
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
        logger.debug(f"Getting genotpes start={start} stop={stop}")

        for variant in self.ts.variants(
            isolated_as_missing=self.isolated_as_missing,
            left=self.positions[start],
            right=self.positions[stop] if stop < self.num_records else None,
            samples=self.tskit_samples,
            copy=False,
        ):
            gt = np.full(shape, constants.INT_FILL, dtype=np.int8)
            alleles = np.full(num_alleles, constants.STR_FILL, dtype="O")
            # length is the length of the REF allele unless other fields
            # are included.
            variant_length = len(variant.alleles[0])
            for i, allele in enumerate(variant.alleles):
                # None is returned by tskit in the case of a missing allele
                if allele is None:
                    continue
                assert i < num_alleles
                alleles[i] = allele
            gt[self.sample_indices, self.ploidy_indices] = variant.genotypes[
                self.genotype_indices
            ]

            yield vcz.VariantData(variant_length, alleles, gt, phased)

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
        dimensions = vcz.standard_dimensions(
            variants_size=m,
            variants_chunk_size=variants_chunk_size,
            samples_size=n,
            samples_chunk_size=samples_chunk_size,
            ploidy_size=self.max_ploidy,
            alleles_size=max_alleles,
        )
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

        tables = self.ts.tables
        ancestral_state_offsets = tables.sites.ancestral_state_offset
        derived_state_offsets = tables.mutations.derived_state_offset
        ancestral_lengths = ancestral_state_offsets[1:] - ancestral_state_offsets[:-1]
        derived_lengths = derived_state_offsets[1:] - derived_state_offsets[:-1]
        max_variant_length = max(
            np.max(ancestral_lengths) if len(ancestral_lengths) > 0 else 0,
            np.max(derived_lengths) if len(derived_lengths) > 0 else 0,
        )

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
                name="variant_length",
                dtype=core.min_int_dtype(0, max_variant_length),
                dimensions=["variants"],
                description="Length of each variant",
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
    ts_or_path,
    vcz_path,
    *,
    model_mapping=None,
    contig_id=None,
    isolated_as_missing=False,
    variants_chunk_size=None,
    samples_chunk_size=None,
    worker_processes=core.DEFAULT_WORKER_PROCESSES,
    show_progress=False,
):
    """
    Convert a :class:`tskit.TreeSequence` (or path to a tree sequence
    file) to VCF Zarr format stored at the specified path.

    .. todo:: Document parameters
    """
    # FIXME there's some tricky details here in how we're handling
    # parallelism that we'll need to tackle properly, and maybe
    # review the current structures a bit. Basically, it looks like
    # we're pickling/unpickling the format object when we have
    # multiple workers, and this results in several copies of the
    # tree sequence object being pass around. This is fine most
    # of the time, but results in lots of memory being used when
    # we're dealing with really massive files.
    # See https://github.com/sgkit-dev/bio2zarr/issues/403
    tskit_format = TskitFormat(
        ts_or_path,
        model_mapping=model_mapping,
        contig_id=contig_id,
        isolated_as_missing=isolated_as_missing,
    )
    schema_instance = tskit_format.generate_schema(
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
    )
    zarr_path = pathlib.Path(vcz_path)
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
