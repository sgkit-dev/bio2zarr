import dataclasses
import logging
import pathlib

import numpy as np
import zarr

from bio2zarr import constants, core, vcz

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PlinkPaths:
    bed_path: str
    bim_path: str
    fam_path: str


class PlinkFormat(vcz.Source):
    @core.requires_optional_dependency("bed_reader", "plink")
    def __init__(self, prefix):
        import bed_reader

        # TODO we will need support multiple chromosomes here to join
        # plinks into on big zarr. So, these will require multiple
        # bed and bim files, but should share a .fam
        self.prefix = str(prefix)
        paths = PlinkPaths(
            self.prefix + ".bed",
            self.prefix + ".bim",
            self.prefix + ".fam",
        )
        self.bed = bed_reader.open_bed(
            paths.bed_path,
            bim_location=paths.bim_path,
            fam_location=paths.fam_path,
            num_threads=1,
            count_A1=True,
        )

    @property
    def path(self):
        return self.prefix

    @property
    def num_records(self):
        return self.bed.sid_count

    @property
    def samples(self):
        return [vcz.Sample(id=sample) for sample in self.bed.iid]

    @property
    def contigs(self):
        return [vcz.Contig(id=str(chrom)) for chrom in np.unique(self.bed.chromosome)]

    @property
    def num_samples(self):
        return len(self.samples)

    def iter_contig(self, start, stop):
        chrom_to_contig_index = {contig.id: i for i, contig in enumerate(self.contigs)}
        for chrom in self.bed.chromosome[start:stop]:
            yield chrom_to_contig_index[str(chrom)]

    def iter_field(self, field_name, shape, start, stop):
        assert field_name == "position"  # Only position field is supported from plink
        yield from self.bed.bp_position[start:stop]

    def iter_id(self, start, stop):
        yield from self.bed.sid[start:stop]

    def iter_alleles_and_genotypes(self, start, stop, shape, num_alleles):
        alt_field = self.bed.allele_1
        ref_field = self.bed.allele_2
        bed_chunk = self.bed.read(slice(start, stop), dtype=np.int8).T
        gt = np.zeros(shape, dtype=np.int8)
        phased = np.zeros(shape[:-1], dtype=bool)
        for i, (ref, alt) in enumerate(
            zip(ref_field[start:stop], alt_field[start:stop])
        ):
            alleles = np.full(num_alleles, constants.STR_FILL, dtype="O")
            alleles[0] = ref
            alleles[1 : 1 + len(alt)] = alt
            gt[:] = 0
            gt[bed_chunk[i] == -127] = -1
            gt[bed_chunk[i] == 2] = 1
            gt[bed_chunk[i] == 1, 0] = 1

            # rlen is the length of the REF in PLINK as there's no END annotations
            yield vcz.VariantData(len(alleles[0]), alleles, gt, phased)

    def generate_schema(
        self,
        variants_chunk_size=None,
        samples_chunk_size=None,
    ):
        n = self.bed.iid_count
        m = self.bed.sid_count
        logging.info(f"Scanned plink with {n} samples and {m} variants")
        dimensions = vcz.standard_dimensions(
            variants_size=m,
            variants_chunk_size=variants_chunk_size,
            samples_size=n,
            samples_chunk_size=samples_chunk_size,
            ploidy_size=2,
            alleles_size=2,
        )
        schema_instance = vcz.VcfZarrSchema(
            format_version=vcz.ZARR_SCHEMA_FORMAT_VERSION,
            dimensions=dimensions,
            fields=[],
        )

        logger.info(
            "Generating schema with chunks="
            f"variants={dimensions['variants'].chunk_size}, "
            f"samples={dimensions['samples'].chunk_size}"
        )
        # If we don't have SVLEN or END annotations, the rlen field is defined
        # as the length of the REF
        max_len = self.bed.allele_2.itemsize

        array_specs = [
            vcz.ZarrArraySpec(
                source="position",
                name="variant_position",
                dtype="i4",
                dimensions=["variants"],
                description=None,
            ),
            vcz.ZarrArraySpec(
                name="variant_allele",
                dtype="O",
                dimensions=["variants", "alleles"],
                description=None,
            ),
            vcz.ZarrArraySpec(
                name="variant_id",
                dtype="O",
                dimensions=["variants"],
                description=None,
            ),
            vcz.ZarrArraySpec(
                name="variant_id_mask",
                dtype="bool",
                dimensions=["variants"],
                description=None,
            ),
            vcz.ZarrArraySpec(
                source=None,
                name="variant_length",
                dtype=core.min_int_dtype(0, max_len),
                dimensions=["variants"],
                description="Length of each variant",
            ),
            vcz.ZarrArraySpec(
                name="variant_contig",
                dtype=core.min_int_dtype(0, len(np.unique(self.bed.chromosome))),
                dimensions=["variants"],
                description="Contig/chromosome index for each variant",
            ),
            vcz.ZarrArraySpec(
                name="call_genotype_phased",
                dtype="bool",
                dimensions=["variants", "samples"],
                description=None,
                compressor=vcz.DEFAULT_ZARR_COMPRESSOR_BOOL.get_config(),
            ),
            vcz.ZarrArraySpec(
                name="call_genotype",
                dtype="i1",
                dimensions=["variants", "samples", "ploidy"],
                description=None,
                compressor=vcz.DEFAULT_ZARR_COMPRESSOR_GENOTYPES.get_config(),
            ),
            vcz.ZarrArraySpec(
                name="call_genotype_mask",
                dtype="bool",
                dimensions=["variants", "samples", "ploidy"],
                description=None,
                compressor=vcz.DEFAULT_ZARR_COMPRESSOR_BOOL.get_config(),
            ),
        ]
        schema_instance.fields = array_specs
        return schema_instance


def convert(
    prefix,
    out,
    *,
    variants_chunk_size=None,
    samples_chunk_size=None,
    worker_processes=1,
    show_progress=False,
):
    plink_format = PlinkFormat(prefix)
    schema_instance = plink_format.generate_schema(
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
    )
    zarr_path = pathlib.Path(out)
    vzw = vcz.VcfZarrWriter(PlinkFormat, zarr_path)
    # Rough heuristic to split work up enough to keep utilisation high
    target_num_partitions = max(1, worker_processes * 4)
    vzw.init(
        plink_format,
        target_num_partitions=target_num_partitions,
        schema=schema_instance,
    )
    vzw.encode_all_partitions(
        worker_processes=worker_processes,
        show_progress=show_progress,
    )
    vzw.finalise(show_progress)
    vzw.create_index()


# FIXME do this more efficiently - currently reading the whole thing
# in for convenience, and also comparing call-by-call
# TODO we should remove this function from the API - it's a test function
# and should be moved into the suite
@core.requires_optional_dependency("bed_reader", "plink")
def validate(bed_path, zarr_path):
    import bed_reader

    root = zarr.open(store=zarr_path, mode="r")
    call_genotype = root["call_genotype"][:]

    bed = bed_reader.open_bed(bed_path + ".bed", count_A1=True, num_threads=1)

    assert call_genotype.shape[0] == bed.sid_count
    assert call_genotype.shape[1] == bed.iid_count
    bed_genotypes = bed.read(dtype="int8").T
    assert call_genotype.shape[0] == bed_genotypes.shape[0]
    assert call_genotype.shape[1] == bed_genotypes.shape[1]
    assert call_genotype.shape[2] == 2

    row_id = 0
    for bed_row, zarr_row in zip(bed_genotypes, call_genotype):
        # print("ROW", row_id)
        # print(bed_row, zarr_row)
        row_id += 1
        for bed_call, zarr_call in zip(bed_row, zarr_row):
            if bed_call == -127:
                assert list(zarr_call) == [-1, -1]
            elif bed_call == 0:
                assert list(zarr_call) == [0, 0]
            elif bed_call == 1:
                assert list(zarr_call) == [1, 0]
            elif bed_call == 2:
                assert list(zarr_call) == [1, 1]
            else:  # pragma no cover
                raise AssertionError(f"Unexpected bed call {bed_call}")
