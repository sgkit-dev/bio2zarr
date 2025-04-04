import contextlib
import logging
import pathlib
import tempfile

from bio2zarr import schema, writer

from . import icf

logger = logging.getLogger(__name__)


def inspect(path):
    path = pathlib.Path(path)
    if not path.exists():
        raise ValueError(f"Path not found: {path}")
    if (path / "metadata.json").exists():
        obj = icf.IntermediateColumnarFormat(path)
    # NOTE: this is too strict, we should support more general Zarrs, see #276
    elif (path / ".zmetadata").exists():
        obj = writer.VcfZarr(path)
    else:
        raise ValueError(f"{path} not in ICF or VCF Zarr format")
    return obj.summary_table()


def mkschema(
    if_path,
    out,
    *,
    variants_chunk_size=None,
    samples_chunk_size=None,
    local_alleles=None,
):
    store = icf.IntermediateColumnarFormat(if_path)
    spec = store.generate_schema(
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
        local_alleles=local_alleles,
    )
    out.write(spec.asjson())


def encode(
    icf_path,
    zarr_path,
    schema_path=None,
    variants_chunk_size=None,
    samples_chunk_size=None,
    max_variant_chunks=None,
    dimension_separator=None,
    max_memory=None,
    local_alleles=None,
    worker_processes=1,
    show_progress=False,
):
    # Rough heuristic to split work up enough to keep utilisation high
    target_num_partitions = max(1, worker_processes * 4)
    encode_init(
        icf_path,
        zarr_path,
        target_num_partitions,
        schema_path=schema_path,
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
        local_alleles=local_alleles,
        max_variant_chunks=max_variant_chunks,
        dimension_separator=dimension_separator,
    )
    vzw = writer.VcfZarrWriter(icf.IntermediateColumnarFormat, zarr_path)
    vzw.encode_all_partitions(
        worker_processes=worker_processes,
        show_progress=show_progress,
        max_memory=max_memory,
    )
    vzw.finalise(show_progress)
    vzw.create_index()


def encode_init(
    icf_path,
    zarr_path,
    target_num_partitions,
    *,
    schema_path=None,
    variants_chunk_size=None,
    samples_chunk_size=None,
    local_alleles=None,
    max_variant_chunks=None,
    dimension_separator=None,
    max_memory=None,
    worker_processes=1,
    show_progress=False,
):
    icf_store = icf.IntermediateColumnarFormat(icf_path)
    if schema_path is None:
        schema_instance = icf_store.generate_schema(
            variants_chunk_size=variants_chunk_size,
            samples_chunk_size=samples_chunk_size,
            local_alleles=local_alleles,
        )
    else:
        logger.info(f"Reading schema from {schema_path}")
        if variants_chunk_size is not None or samples_chunk_size is not None:
            raise ValueError(
                "Cannot specify schema along with chunk sizes"
            )  # NEEDS TEST
        with open(schema_path) as f:
            schema_instance = schema.VcfZarrSchema.fromjson(f.read())
    zarr_path = pathlib.Path(zarr_path)
    vzw = writer.VcfZarrWriter("icf", zarr_path)
    return vzw.init(
        icf_store,
        target_num_partitions=target_num_partitions,
        schema=schema_instance,
        dimension_separator=dimension_separator,
        max_variant_chunks=max_variant_chunks,
    )


def encode_partition(zarr_path, partition):
    writer_instance = writer.VcfZarrWriter(icf.IntermediateColumnarFormat, zarr_path)
    writer_instance.encode_partition(partition)


def encode_finalise(zarr_path, show_progress=False):
    writer_instance = writer.VcfZarrWriter(icf.IntermediateColumnarFormat, zarr_path)
    writer_instance.finalise(show_progress=show_progress)


def convert(
    vcfs,
    out_path,
    *,
    variants_chunk_size=None,
    samples_chunk_size=None,
    worker_processes=1,
    local_alleles=None,
    show_progress=False,
    icf_path=None,
):
    if icf_path is None:
        cm = temp_icf_path(prefix="vcf2zarr")
    else:
        cm = contextlib.nullcontext(icf_path)

    with cm as icf_path:
        icf.explode(
            icf_path,
            vcfs,
            worker_processes=worker_processes,
            show_progress=show_progress,
        )
        encode(
            icf_path,
            out_path,
            variants_chunk_size=variants_chunk_size,
            samples_chunk_size=samples_chunk_size,
            worker_processes=worker_processes,
            show_progress=show_progress,
            local_alleles=local_alleles,
        )


@contextlib.contextmanager
def temp_icf_path(prefix=None):
    with tempfile.TemporaryDirectory(prefix=prefix) as tmp:
        yield pathlib.Path(tmp) / "icf"
