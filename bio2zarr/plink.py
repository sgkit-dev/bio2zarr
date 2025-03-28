import logging

import bed_reader
import humanfriendly
import numpy as np
import zarr

from bio2zarr import schema, writer
from bio2zarr.zarr_utils import ZARR_FORMAT_KWARGS

from . import core

logger = logging.getLogger(__name__)


def generate_schema(bed_path, variants_chunk_size=None, samples_chunk_size=None):
    """
    Generate a schema for PLINK data based on the contents of the bed file.
    """
    bed = bed_reader.open_bed(bed_path, num_threads=1)
    n = bed.iid_count
    m = bed.sid_count

    if samples_chunk_size is None:
        samples_chunk_size = 1000
    if variants_chunk_size is None:
        variants_chunk_size = 10_000

    logger.info(
        f"Generating PLINK schema with chunks={variants_chunk_size, samples_chunk_size}"
    )

    ploidy = 2
    shape = [m, n]
    chunks = [variants_chunk_size, samples_chunk_size]
    dimensions = ["variants", "samples"]

    array_specs = [
        # Sample information
        schema.ZarrArraySpec.new(
            vcf_field=None,
            name="sample_id",
            dtype="O",
            shape=(n,),
            chunks=(samples_chunk_size,),
            dimensions=["samples"],
            description="Sample identifiers",
        ),
        # Variant information
        schema.ZarrArraySpec.new(
            vcf_field=None,
            name="variant_position",
            dtype=np.int32,
            shape=(m,),
            chunks=(variants_chunk_size,),
            dimensions=["variants"],
            description="The reference position",
        ),
        schema.ZarrArraySpec.new(
            vcf_field=None,
            name="variant_allele",
            dtype="O",
            shape=(m, 2),
            chunks=(variants_chunk_size, 2),
            dimensions=["variants", "alleles"],
            description="List of the reference and alternate alleles",
        ),
        # Genotype information
        schema.ZarrArraySpec.new(
            vcf_field=None,
            name="call_genotype_phased",
            dtype="bool",
            shape=list(shape),
            chunks=list(chunks),
            dimensions=list(dimensions),
            description="Boolean flag indicating if genotypes are phased",
        ),
    ]

    # Add ploidy dimension for genotype arrays
    shape_with_ploidy = shape + [ploidy]
    chunks_with_ploidy = chunks + [ploidy]
    dimensions_with_ploidy = dimensions + ["ploidy"]

    array_specs.extend(
        [
            schema.ZarrArraySpec.new(
                vcf_field=None,
                name="call_genotype",
                dtype="i1",
                shape=list(shape_with_ploidy),
                chunks=list(chunks_with_ploidy),
                dimensions=list(dimensions_with_ploidy),
                description="Genotype calls coded as allele indices",
            ),
            schema.ZarrArraySpec.new(
                vcf_field=None,
                name="call_genotype_mask",
                dtype="bool",
                shape=list(shape_with_ploidy),
                chunks=list(chunks_with_ploidy),
                dimensions=list(dimensions_with_ploidy),
                description="Mask indicating missing genotype calls",
            ),
        ]
    )

    # Create empty lists for VCF-specific metadata
    samples = [{"id": sample_id} for sample_id in bed.iid]
    contigs = []  # PLINK doesn't have contig information in the same way as VCF
    filters = []  # PLINK doesn't use filters like VCF

    return schema.VcfZarrSchema(
        format_version=schema.ZARR_SCHEMA_FORMAT_VERSION,
        samples_chunk_size=samples_chunk_size,
        variants_chunk_size=variants_chunk_size,
        fields=array_specs,
        samples=samples,
        contigs=contigs,
        filters=filters,
    )


def encode_genotypes_slice(bed_path, zarr_path, start, stop):
    # We need to count the A2 alleles here if we want to keep the
    # alleles reported as allele_1, allele_2. It's obvious here what
    # the correct approach is, but it is important to note that the
    # 0th allele is *not* necessarily the REF for these datasets.
    bed = bed_reader.open_bed(bed_path, num_threads=1, count_A1=False)
    root = zarr.open(store=zarr_path, mode="a", **ZARR_FORMAT_KWARGS)
    gt = core.BufferedArray(root["call_genotype"], start)
    gt_mask = core.BufferedArray(root["call_genotype_mask"], start)
    gt_phased = core.BufferedArray(root["call_genotype_phased"], start)
    variants_chunk_size = gt.array.chunks[0]
    assert start % variants_chunk_size == 0

    logger.debug(f"Reading slice {start}:{stop}")
    chunk_start = start
    while chunk_start < stop:
        chunk_stop = min(chunk_start + variants_chunk_size, stop)
        logger.debug(f"Reading bed slice {chunk_start}:{chunk_stop}")
        bed_chunk = bed.read(slice(chunk_start, chunk_stop), dtype=np.int8).T
        logger.debug(f"Got bed slice {humanfriendly.format_size(bed_chunk.nbytes)}")
        # Probably should do this without iterating over rows, but it's a bit
        # simpler and lines up better with the array buffering API. The bottleneck
        # is in the encoding anyway.
        for values in bed_chunk:
            j = gt.next_buffer_row()
            g = np.zeros_like(gt.buff[j])
            g[values == -127] = -1
            g[values == 2] = 1
            g[values == 1, 0] = 1
            gt.buff[j] = g
            j = gt_phased.next_buffer_row()
            gt_phased.buff[j] = False
            j = gt_mask.next_buffer_row()
            gt_mask.buff[j] = gt.buff[j] == -1
        chunk_start = chunk_stop
    gt.flush()
    gt_phased.flush()
    gt_mask.flush()
    logger.debug(f"GT slice {start}:{stop} done")


def convert(
    bed_path,
    zarr_path,
    *,
    show_progress=False,
    worker_processes=1,
    variants_chunk_size=None,
    samples_chunk_size=None,
):
    """
    Convert PLINK data to zarr format using the shared writer infrastructure.
    """
    # Generate schema from the PLINK data
    plink_schema = generate_schema(
        bed_path,
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
    )

    # Create a data source adapter for PLINK
    plink_adapter = PlinkDataAdapter(bed_path)

    # Use the general writer
    writer_instance = writer.GenericZarrWriter(zarr_path)
    writer_instance.init_from_schema(plink_schema)

    # Encode data using the writer
    logger.info(f"Converting PLINK data to zarr at {zarr_path}")
    writer_instance.encode_data(
        plink_adapter, worker_processes=worker_processes, show_progress=show_progress
    )

    # Finalize the zarr store
    writer_instance.finalise(show_progress)
    zarr.consolidate_metadata(zarr_path)
    logger.info("PLINK conversion complete")


class PlinkDataAdapter:
    """
    Adapter class to provide PLINK data to the generic writer.
    """

    def __init__(self, bed_path):
        self.bed_path = bed_path
        self.bed = bed_reader.open_bed(bed_path, num_threads=1)
        self.n_samples = self.bed.iid_count
        self.n_variants = self.bed.sid_count

    def get_sample_ids(self):
        return self.bed.iid

    def get_variant_positions(self):
        return self.bed.bp_position

    def get_variant_alleles(self):
        return np.stack([self.bed.allele_1, self.bed.allele_2], axis=1)

    def get_genotypes_slice(self, start, stop):
        """
        Read a slice of genotypes from the PLINK data.
        Returns a dictionary with three arrays:
        - genotypes: The actual genotype values
        - phased: Whether genotypes are phased (always False for PLINK)
        - mask: Which genotype values are missing
        """
        bed_chunk = self.bed.read(slice(start, stop), dtype=np.int8).T
        n_variants = stop - start

        # Create return arrays
        gt = np.zeros((n_variants, self.n_samples, 2), dtype=np.int8)
        gt_phased = np.zeros((n_variants, self.n_samples), dtype=bool)
        gt_mask = np.zeros((n_variants, self.n_samples, 2), dtype=bool)

        # Convert PLINK encoding to genotype encoding
        # PLINK: 0=hom ref, 1=het, 2=hom alt, -127=missing
        # Zarr: [0,0]=hom ref, [1,0]=het, [1,1]=hom alt, [-1,-1]=missing
        for i, values in enumerate(bed_chunk):
            gt[i, values == -127] = -1
            gt[i, values == 2, :] = 1
            gt[i, values == 1, 0] = 1
            gt_mask[i] = gt[i] == -1

        return {
            "call_genotype": gt,
            "call_genotype_phased": gt_phased,
            "call_genotype_mask": gt_mask,
        }

    def close(self):
        del self.bed


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
