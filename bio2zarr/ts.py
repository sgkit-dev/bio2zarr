import logging
import pathlib

import numpy as np
import tskit

from bio2zarr import constants, core, vcz

logger = logging.getLogger(__name__)


class TskitFormat:
    def __init__(self, ts_path, contig_id=None, ploidy=None, isolated_as_missing=False):
        self.path = ts_path
        self.ts = tskit.load(ts_path)
        self.contig_id = contig_id if contig_id is not None else "1"
        self.isolated_as_missing = isolated_as_missing
        self.root_attrs = {}

        self._make_sample_mapping(ploidy)
        self.contigs = [vcz.Contig(id=self.contig_id)]
        self.num_records = self.ts.num_sites
        self.positions = self.ts.sites_position

    def _make_sample_mapping(self, ploidy):
        ts = self.ts
        self.individual_ploidies = []
        self.max_ploidy = 0

        if ts.num_individuals > 0 and ploidy is not None:
            raise ValueError(
                "Cannot specify ploidy when individuals are present in tables"
            )

        # Find all sample nodes that reference individuals
        individuals = np.unique(ts.tables.nodes.individual[ts.samples()])
        if len(individuals) == 1 and individuals[0] == tskit.NULL:
            # No samples refer to individuals
            individuals = None
        else:
            # np.unique sorts the argument, so if NULL (-1) is present it
            # will be the first value.
            if individuals[0] == tskit.NULL:
                raise ValueError(
                    "Sample nodes must either all be associated with individuals "
                    "or not associated with any individuals"
                )

        if individuals is not None:
            self.sample_ids = []
            for i in individuals:
                if i < 0 or i >= self.ts.num_individuals:
                    raise ValueError("Invalid individual IDs provided.")
                ind = self.ts.individual(i)
                if len(ind.nodes) == 0:
                    raise ValueError(f"Individual {i} not associated with a node")
                is_sample = {ts.node(u).is_sample() for u in ind.nodes}
                if len(is_sample) != 1:
                    raise ValueError(
                        f"Individual {ind.id} has nodes that are sample and "
                        "non-samples"
                    )
                self.sample_ids.extend(ind.nodes)
                self.individual_ploidies.append(len(ind.nodes))
                self.max_ploidy = max(self.max_ploidy, len(ind.nodes))
        else:
            if ploidy is None:
                ploidy = 1
            if ploidy < 1:
                raise ValueError("Ploidy must be >= 1")
            if ts.num_samples % ploidy != 0:
                raise ValueError("Sample size must be divisible by ploidy")
            self.individual_ploidies = np.full(
                ts.num_samples // ploidy, ploidy, dtype=np.int32
            )
            self.max_ploidy = ploidy
            self.sample_ids = np.arange(ts.num_samples, dtype=np.int32)

        self.num_samples = len(self.individual_ploidies)

        self.samples = [vcz.Sample(id=f"tsk_{j}") for j in range(self.num_samples)]

    def iter_alleles(self, start, stop, num_alleles):
        for variant in self.ts.variants(
            samples=self.sample_ids,
            isolated_as_missing=self.isolated_as_missing,
            left=self.positions[start],
            right=self.positions[stop] if stop < self.num_records else None,
        ):
            alleles = np.full(num_alleles, constants.STR_FILL, dtype="O")
            for i, allele in enumerate(variant.alleles):
                assert i < num_alleles
                alleles[i] = allele
            yield alleles

    def iter_contig(self, start, stop):
        yield from (0 for _ in range(start, stop))

    def iter_field(self, field_name, shape, start, stop):
        if field_name == "position":
            for pos in self.ts.tables.sites.position[start:stop]:
                yield int(pos)
        else:
            raise ValueError(f"Unknown field {field_name}")

    def iter_genotypes(self, shape, start, stop):
        gt = np.zeros(shape, dtype=np.int8)
        phased = np.zeros(shape[:-1], dtype=bool)

        for variant in self.ts.variants(
            samples=self.sample_ids,
            isolated_as_missing=self.isolated_as_missing,
            left=self.positions[start],
            right=self.positions[stop] if stop < self.num_records else None,
        ):
            genotypes = variant.genotypes

            sample_index = 0
            for i, ploidy in enumerate(self.individual_ploidies):
                for j in range(ploidy):
                    if j < self.max_ploidy:  # Only fill up to max_ploidy
                        try:
                            gt[i, j] = genotypes[sample_index + j]
                        except IndexError:
                            # This can happen if the ploidy varies between individuals
                            gt[i, j] = -2  # Fill value

                # In tskit, all genotypes are considered phased
                phased[i] = True
                sample_index += ploidy

            yield gt, phased

    def generate_schema(
        self,
        variants_chunk_size=None,
        samples_chunk_size=None,
    ):
        n = self.num_samples
        m = self.ts.num_sites

        # Determine max number of alleles
        max_alleles = 0
        for variant in self.ts.variants():
            max_alleles = max(max_alleles, len(variant.alleles))

        logging.info(f"Scanned tskit with {n} samples and {m} variants")
        logging.info(
            f"Maximum ploidy: {self.max_ploidy}, maximum alleles: {max_alleles}"
        )

        schema_instance = vcz.VcfZarrSchema(
            format_version=vcz.ZARR_SCHEMA_FORMAT_VERSION,
            samples_chunk_size=samples_chunk_size,
            variants_chunk_size=variants_chunk_size,
            fields=[],
        )

        logger.info(
            "Generating schema with chunks="
            f"{schema_instance.variants_chunk_size, schema_instance.samples_chunk_size}"
        )

        array_specs = [
            vcz.ZarrArraySpec.new(
                vcf_field="position",
                name="variant_position",
                dtype="i4",
                shape=[m],
                dimensions=["variants"],
                chunks=[schema_instance.variants_chunk_size],
                description="Position of each variant",
            ),
            vcz.ZarrArraySpec.new(
                vcf_field=None,
                name="variant_allele",
                dtype="O",
                shape=[m, max_alleles],
                dimensions=["variants", "alleles"],
                chunks=[schema_instance.variants_chunk_size, max_alleles],
                description="Alleles for each variant",
            ),
            vcz.ZarrArraySpec.new(
                vcf_field=None,
                name="variant_contig",
                dtype=core.min_int_dtype(0, len(self.contigs)),
                shape=[m],
                dimensions=["variants"],
                chunks=[schema_instance.variants_chunk_size],
                description="Contig/chromosome index for each variant",
            ),
            vcz.ZarrArraySpec.new(
                vcf_field=None,
                name="call_genotype_phased",
                dtype="bool",
                shape=[m, n],
                dimensions=["variants", "samples"],
                chunks=[
                    schema_instance.variants_chunk_size,
                    schema_instance.samples_chunk_size,
                ],
                description="Whether the genotype is phased",
            ),
            vcz.ZarrArraySpec.new(
                vcf_field=None,
                name="call_genotype",
                dtype="i1",
                shape=[m, n, self.max_ploidy],
                dimensions=["variants", "samples", "ploidy"],
                chunks=[
                    schema_instance.variants_chunk_size,
                    schema_instance.samples_chunk_size,
                    self.max_ploidy,
                ],
                description="Genotype for each variant and sample",
            ),
            vcz.ZarrArraySpec.new(
                vcf_field=None,
                name="call_genotype_mask",
                dtype="bool",
                shape=[m, n, self.max_ploidy],
                dimensions=["variants", "samples", "ploidy"],
                chunks=[
                    schema_instance.variants_chunk_size,
                    schema_instance.samples_chunk_size,
                    self.max_ploidy,
                ],
                description="Mask for each genotype call",
            ),
        ]
        schema_instance.fields = array_specs
        return schema_instance


def convert(
    ts_path,
    zarr_path,
    *,
    contig_id=None,
    ploidy=None,
    isolated_as_missing=False,
    variants_chunk_size=None,
    samples_chunk_size=None,
    worker_processes=1,
    show_progress=False,
):
    tskit_format = TskitFormat(
        ts_path,
        contig_id=contig_id,
        ploidy=ploidy,
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
