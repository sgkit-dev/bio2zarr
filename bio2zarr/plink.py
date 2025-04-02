import logging
import pathlib

import bed_reader
import numpy as np
import zarr

logger = logging.getLogger(__name__)


class PlinkFormat:
    def __init__(self, path):
        self.path = path
        self.bed = bed_reader.open_bed(path, num_threads=1)
        self.num_records = self.bed.sid_count
        self.samples = list(self.bed.iid)
        self.num_samples = len(self.samples)
        self.root_attrs = {}

    def iter_alleles(self, start, stop, num_alleles):
        ref_field = self.bed.allele_1
        alt_field = self.bed.allele_2

        # TODO - should be doing whole chunks rather than one at a time
        for ref, alt in zip(
            ref_field[start:stop],
            alt_field[start:stop],
        ):
            alleles = np.full(num_alleles, constants.STR_FILL, dtype="O")
            alleles[0] = ref
            alleles[1 : 1 + len(alt)] = alt
            yield alleles

    def iter_field(self, field_name, shape, start, stop):
        data = {
            "position": self.bed.bp_position,
        }[field_name]
        yield from data[start:stop]

    def iter_genotypes(self, shape, start, stop):
        bed_chunk = self.bed.read(slice(start, stop), dtype=np.int8).T
        gt = np.zeros(shape, dtype=np.int8)
        phased = np.zeros(shape[:-1], dtype=bool)
        for values in bed_chunk:
            gt[values == -127] = -1  # Missing values
            gt[values == 0] = [1, 1]  # Homozygous ALT (2 in PLINK)
            gt[values == 1] = [1, 0]  # Heterozygous (1 in PLINK)
            gt[values == 2] = [0, 0]  # Homozygous REF (0 in PLINK)
            yield gt, phased


# Import here to avoid circular import
from bio2zarr import constants, schema, writer  # noqa: E402


def generate_schema(
    bed,
    variants_chunk_size=None,
    samples_chunk_size=None,
):
    n = bed.iid_count
    m = bed.sid_count
    logging.info(f"Scanned plink with {n} samples and {m} variants")

    # FIXME
    if samples_chunk_size is None:
        samples_chunk_size = 1000
    if variants_chunk_size is None:
        variants_chunk_size = 10_000

    logger.info(
        f"Generating schema with chunks={variants_chunk_size, samples_chunk_size}"
    )

    array_specs = [
        schema.ZarrArraySpec.new(
            vcf_field="position",
            name="variant_position",
            dtype="i4",
            shape=[m],
            dimensions=["variants"],
            chunks=[variants_chunk_size],
            description=None,
        ),
        schema.ZarrArraySpec.new(
            vcf_field=None,
            name="variant_allele",
            dtype="str",
            shape=[m, 2],
            dimensions=["variants", "alleles"],
            chunks=[variants_chunk_size, 2],
            description=None,
        ),
        schema.ZarrArraySpec.new(
            vcf_field=None,
            name="call_genotype_phased",
            dtype="bool",
            shape=[m, n],
            dimensions=["variants", "samples"],
            chunks=[variants_chunk_size, samples_chunk_size],
            description=None,
        ),
        schema.ZarrArraySpec.new(
            vcf_field=None,
            name="call_genotype",
            dtype="i1",
            shape=[m, n, 2],
            dimensions=["variants", "samples", "ploidy"],
            chunks=[variants_chunk_size, samples_chunk_size, 2],
            description=None,
        ),
        schema.ZarrArraySpec.new(
            vcf_field=None,
            name="call_genotype_mask",
            dtype="bool",
            shape=[m, n, 2],
            dimensions=["variants", "samples", "ploidy"],
            chunks=[variants_chunk_size, samples_chunk_size, 2],
            description=None,
        ),
    ]

    return schema.VcfZarrSchema(
        format_version=schema.ZARR_SCHEMA_FORMAT_VERSION,
        samples_chunk_size=samples_chunk_size,
        variants_chunk_size=variants_chunk_size,
        fields=array_specs,
        samples=[schema.Sample(id=sample) for sample in bed.iid],
        contigs=[],
        filters=[],
    )


def convert(
    bed_path,
    zarr_path,
    *,
    variants_chunk_size=None,
    samples_chunk_size=None,
    worker_processes=1,
    show_progress=False,
):
    bed = bed_reader.open_bed(bed_path, num_threads=1)
    schema_instance = generate_schema(
        bed,
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
    )
    zarr_path = pathlib.Path(zarr_path)
    vzw = writer.VcfZarrWriter("plink", zarr_path)
    # Rough heuristic to split work up enough to keep utilisation high
    target_num_partitions = max(1, worker_processes * 4)
    vzw.init(
        PlinkFormat(bed_path),
        target_num_partitions=target_num_partitions,
        schema=schema_instance,
        #        dimension_separator=None,
        #       max_variant_chunks=None
    )
    vzw.encode_all_partitions(
        worker_processes=worker_processes,
        show_progress=show_progress,
        # max_memory=None,
    )
    vzw.finalise(show_progress)

    # TODO - index code needs variant_contig
    # vzw.create_index()


# def encode_genotypes_slice(bed_path, zarr_path, start, stop):
#     # We need to count the A2 alleles here if we want to keep the
#     # alleles reported as allele_1, allele_2. It's obvious here what
#     # the correct approach is, but it is important to note that the
#     # 0th allele is *not* necessarily the REF for these datasets.
#     bed = bed_reader.open_bed(bed_path, num_threads=1, count_A1=False)
#     root = zarr.open(store=zarr_path, mode="a", **ZARR_FORMAT_KWARGS)
#     gt = core.BufferedArray(root["call_genotype"], start)
#     gt_mask = core.BufferedArray(root["call_genotype_mask"], start)
#     gt_phased = core.BufferedArray(root["call_genotype_phased"], start)
#     variants_chunk_size = gt.array.chunks[0]
#     assert start % variants_chunk_size == 0

#     logger.debug(f"Reading slice {start}:{stop}")
#     chunk_start = start
#     while chunk_start < stop:
#         chunk_stop = min(chunk_start + variants_chunk_size, stop)
#         logger.debug(f"Reading bed slice {chunk_start}:{chunk_stop}")
#         bed_chunk = bed.read(slice(chunk_start, chunk_stop), dtype=np.int8).T
#         logger.debug(f"Got bed slice {humanfriendly.format_size(bed_chunk.nbytes)}")
#         # Probably should do this without iterating over rows, but it's a bit
#         # simpler and lines up better with the array buffering API. The bottleneck
#         # is in the encoding anyway.
#         for values in bed_chunk:
#             j = gt.next_buffer_row()
#             g = np.zeros_like(gt.buff[j])
#             g[values == -127] = -1
#             g[values == 2] = 1
#             g[values == 1, 0] = 1
#             gt.buff[j] = g
#             j = gt_phased.next_buffer_row()
#             gt_phased.buff[j] = False
#             j = gt_mask.next_buffer_row()
#             gt_mask.buff[j] = gt.buff[j] == -1
#         chunk_start = chunk_stop
#     gt.flush()
#     gt_phased.flush()
#     gt_mask.flush()
#     logger.debug(f"GT slice {start}:{stop} done")

# root = zarr.open_group(store=zarr_path, mode="w", **ZARR_FORMAT_KWARGS)

# ploidy = 2
# shape = [m, n]
# chunks = [variants_chunk_size, samples_chunk_size]
# dimensions = ["variants", "samples"]

# # TODO we should be reusing some logic from vcfzarr here on laying
# # out the basic dataset, and using the schema generator. Currently
# # we're not using the best Blosc settings for genotypes here.
# default_compressor = numcodecs.Blosc(cname="zstd", clevel=7)

# a = root.array(
#     "sample_id",
#     data=bed.iid,
#     shape=bed.iid.shape,
#     dtype="str",
#     compressor=default_compressor,
#     chunks=(samples_chunk_size,),
# )
# a.attrs["_ARRAY_DIMENSIONS"] = ["samples"]
# logger.debug("Encoded samples")

# # TODO encode these in slices - but read them in one go to avoid
# # fetching repeatedly from bim file
# a = root.array(
#     "variant_position",
#     data=bed.bp_position,
#     shape=bed.bp_position.shape,
#     dtype=np.int32,
#     compressor=default_compressor,
#     chunks=(variants_chunk_size,),
# )
# a.attrs["_ARRAY_DIMENSIONS"] = ["variants"]
# logger.debug("encoded variant_position")

# alleles = np.stack([bed.allele_1, bed.allele_2], axis=1)
# a = root.array(
#     "variant_allele",
#     data=alleles,
#     shape=alleles.shape,
#     dtype="str",
#     compressor=default_compressor,
#     chunks=(variants_chunk_size, alleles.shape[1]),
# )
# a.attrs["_ARRAY_DIMENSIONS"] = ["variants", "alleles"]
# logger.debug("encoded variant_allele")

# # TODO remove this?
# a = root.empty(
#     name="call_genotype_phased",
#     dtype="bool",
#     shape=list(shape),
#     chunks=list(chunks),
#     compressor=default_compressor,
#     **ZARR_FORMAT_KWARGS,
# )
# a.attrs["_ARRAY_DIMENSIONS"] = list(dimensions)

# shape += [ploidy]
# dimensions += ["ploidy"]
# a = root.empty(
#     name="call_genotype",
#     dtype="i1",
#     shape=list(shape),
#     chunks=list(chunks),
#     compressor=default_compressor,
#     **ZARR_FORMAT_KWARGS,
# )
# a.attrs["_ARRAY_DIMENSIONS"] = list(dimensions)

# a = root.empty(
#     name="call_genotype_mask",
#     dtype="bool",
#     shape=list(shape),
#     chunks=list(chunks),
#     compressor=default_compressor,
#     **ZARR_FORMAT_KWARGS,
# )
# a.attrs["_ARRAY_DIMENSIONS"] = list(dimensions)

# del bed

# num_slices = max(1, worker_processes * 4)
# slices = core.chunk_aligned_slices(a, num_slices)

# total_chunks = sum(a.nchunks for _, a in root.arrays())

# progress_config = core.ProgressConfig(
#     total=total_chunks, title="Convert", units="chunks", show=show_progress
# )
# with core.ParallelWorkManager(worker_processes, progress_config) as pwm:
#     for start, stop in slices:
#         pwm.submit(encode_genotypes_slice, bed_path, zarr_path, start, stop)

# # TODO also add atomic swap like VCF. Should be abstracted to
# # share basic code for setting up the variation dataset zarr
# zarr.consolidate_metadata(zarr_path)


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
