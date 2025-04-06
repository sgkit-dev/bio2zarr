import logging
import pathlib

import bed_reader
import numpy as np
import zarr

from bio2zarr import constants, core, vcz

logger = logging.getLogger(__name__)


class PlinkFormat(vcz.Source):
    def __init__(self, path):
        self._path = pathlib.Path(path)
        self.bed = bed_reader.open_bed(path, num_threads=1, count_A1=False)

    @property
    def path(self):
        return self._path

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

    def iter_alleles_and_genotypes(self, start, stop, shape, num_alleles):
        ref_field = self.bed.allele_1
        alt_field = self.bed.allele_2
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

            yield alleles, (gt, phased)

    def generate_schema(
        self,
        variants_chunk_size=None,
        samples_chunk_size=None,
    ):
        n = self.bed.iid_count
        m = self.bed.sid_count
        logging.info(f"Scanned plink with {n} samples and {m} variants")

        # Define dimensions with sizes and chunk sizes
        dimensions = {
            "variants": vcz.VcfZarrDimension(
                size=m, chunk_size=variants_chunk_size or vcz.DEFAULT_VARIANT_CHUNK_SIZE
            ),
            "samples": vcz.VcfZarrDimension(
                size=n, chunk_size=samples_chunk_size or vcz.DEFAULT_SAMPLE_CHUNK_SIZE
            ),
            "ploidy": vcz.VcfZarrDimension(size=2),
            "alleles": vcz.VcfZarrDimension(size=2),
        }

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
                compressor=vcz.DEFAULT_ZARR_COMPRESSOR_BOOL.get_config(),
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
    bed_path,
    zarr_path,
    *,
    variants_chunk_size=None,
    samples_chunk_size=None,
    worker_processes=1,
    show_progress=False,
):
    plink_format = PlinkFormat(bed_path)
    schema_instance = plink_format.generate_schema(
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
    )
    zarr_path = pathlib.Path(zarr_path)
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
def validate(bed_path, zarr_path):
    root = zarr.open(store=zarr_path, mode="r")
    call_genotype = root["call_genotype"][:]

    bed = bed_reader.open_bed(bed_path, count_A1=False, num_threads=1)

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
