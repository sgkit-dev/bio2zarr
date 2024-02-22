import numpy as np
import zarr
import bed_reader

from . import core


def encode_bed_partition_genotypes(
    bed_path, zarr_path, start_variant, end_variant, encoder_threads=8
):
    bed = bed_reader.open_bed(bed_path, num_threads=1)

    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store=store)
    gt = core.BufferedArray(root["call_genotype"])
    gt_mask = core.BufferedArray(root["call_genotype_mask"])
    gt_phased = core.BufferedArray(root["call_genotype_phased"])
    chunk_length = gt.array.chunks[0]
    assert start_variant % chunk_length == 0

    buffered_arrays = [gt, gt_phased, gt_mask]

    with core.ThreadedZarrEncoder(buffered_arrays, encoder_threads) as te:
        start = start_variant
        while start < end_variant:
            stop = min(start + chunk_length, end_variant)
            bed_chunk = bed.read(index=slice(start, stop), dtype="int8").T
            # Note could do this without iterating over rows, but it's a bit
            # simpler and the bottleneck is in the encoding step anyway. It's
            # also nice to have updates on the progress monitor.
            for values in bed_chunk:
                j = te.next_buffer_row()
                dest = gt.buff[j]
                dest[values == -127] = -1
                dest[values == 2] = 1
                dest[values == 1, 0] = 1
                gt_phased.buff[j] = False
                gt_mask.buff[j] = dest == -1
                core.update_progress(1)
            start = stop


def convert(
    bed_path,
    zarr_path,
    *,
    show_progress=False,
    worker_processes=1,
    chunk_length=None,
    chunk_width=None,
):
    bed = bed_reader.open_bed(bed_path, num_threads=1)
    n = bed.iid_count
    m = bed.sid_count
    del bed

    # FIXME
    if chunk_width is None:
        chunk_width = 1000
    if chunk_length is None:
        chunk_length = 10_000

    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store=store, overwrite=True)

    ploidy = 2
    shape = [m, n]
    chunks = [chunk_length, chunk_width]
    dimensions = ["variants", "samples"]

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
        dtype="i8",
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

    chunks_per_future = 2  # FIXME - make a parameter
    start = 0
    partitions = []
    while start < m:
        stop = min(m, start + chunk_length * chunks_per_future)
        partitions.append((start, stop))
        start = stop
    assert start == m

    progress_config = core.ProgressConfig(
        total=m, title="Convert", units="vars", show=show_progress
    )
    with core.ParallelWorkManager(worker_processes, progress_config) as pwm:
        for start, end in partitions:
            pwm.submit(encode_bed_partition_genotypes, bed_path, zarr_path, start, end)

    # TODO also add atomic swap like VCF. Should be abstracted to
    # share basic code for setting up the variation dataset zarr
    zarr.consolidate_metadata(zarr_path)
