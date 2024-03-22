import logging

import humanfriendly
import numpy as np
import zarr
import bed_reader

from . import core


logger = logging.getLogger(__name__)


def encode_genotypes_slice(bed_path, zarr_path, start, stop):
    # We need to count the A2 alleles here if we want to keep the
    # alleles reported as allele_1, allele_2. It's obvious here what
    # the correct approach is, but it is important to note that the
    # 0th allele is *not* necessarily the REF for these datasets.
    bed = bed_reader.open_bed(bed_path, num_threads=1, count_A1=False)
    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store=store)
    gt = core.BufferedArray(root["call_genotype"], start)
    gt_mask = core.BufferedArray(root["call_genotype_mask"], start)
    gt_phased = core.BufferedArray(root["call_genotype_phased"], start)
    variants_chunk_size = gt.array.chunks[0]
    n = gt.array.shape[1]
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
    bed = bed_reader.open_bed(bed_path, num_threads=1)
    n = bed.iid_count
    m = bed.sid_count
    logging.info(f"Scanned plink with {n} samples and {m} variants")

    # FIXME
    if samples_chunk_size is None:
        samples_chunk_size = 1000
    if variants_chunk_size is None:
        variants_chunk_size = 10_000

    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store=store, overwrite=True)

    ploidy = 2
    shape = [m, n]
    chunks = [variants_chunk_size, samples_chunk_size]
    dimensions = ["variants", "samples"]

    a = root.array(
        "sample_id",
        bed.iid,
        dtype="str",
        compressor=core.default_compressor,
        chunks=(samples_chunk_size,),
    )
    a.attrs["_ARRAY_DIMENSIONS"] = ["samples"]
    logger.debug(f"Encoded samples")

    # TODO encode these in slices - but read them in one go to avoid
    # fetching repeatedly from bim file
    a = root.array(
        "variant_position",
        bed.bp_position,
        dtype=np.int32,
        compressor=core.default_compressor,
        chunks=(variants_chunk_size,),
    )
    a.attrs["_ARRAY_DIMENSIONS"] = ["variants"]
    logger.debug(f"encoded variant_position")

    alleles = np.stack([bed.allele_1, bed.allele_2], axis=1)
    a = root.array(
        "variant_allele",
        alleles,
        dtype="str",
        compressor=core.default_compressor,
        chunks=(variants_chunk_size,),
    )
    a.attrs["_ARRAY_DIMENSIONS"] = ["variants", "alleles"]
    logger.debug(f"encoded variant_allele")

    # TODO remove this?
    a = root.empty(
        "call_genotype_phased",
        dtype="bool",
        shape=list(shape),
        chunks=list(chunks),
        compressor=core.default_compressor,
    )
    a.attrs["_ARRAY_DIMENSIONS"] = list(dimensions)

    shape += [ploidy]
    dimensions += ["ploidy"]
    a = root.empty(
        "call_genotype",
        dtype="i1",
        shape=list(shape),
        chunks=list(chunks),
        compressor=core.default_compressor,
    )
    a.attrs["_ARRAY_DIMENSIONS"] = list(dimensions)

    a = root.empty(
        "call_genotype_mask",
        dtype="bool",
        shape=list(shape),
        chunks=list(chunks),
        compressor=core.default_compressor,
    )
    a.attrs["_ARRAY_DIMENSIONS"] = list(dimensions)

    del bed

    num_slices = max(1, worker_processes * 4)
    slices = core.chunk_aligned_slices(a, num_slices)

    total_chunks = sum(a.nchunks for a in root.values())

    progress_config = core.ProgressConfig(
        total=total_chunks, title="Convert", units="chunks", show=show_progress
    )
    with core.ParallelWorkManager(worker_processes, progress_config) as pwm:
        for start, stop in slices:
            pwm.submit(encode_genotypes_slice, bed_path, zarr_path, start, stop)

    # TODO also add atomic swap like VCF. Should be abstracted to
    # share basic code for setting up the variation dataset zarr
    zarr.consolidate_metadata(zarr_path)


# FIXME do this more efficiently - currently reading the whole thing
# in for convenience, and also comparing call-by-call
def validate(bed_path, zarr_path):
    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store=store)
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
                assert False
