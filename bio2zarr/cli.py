import click
import tabulate
import coloredlogs

from . import vcf
from . import vcf_utils
from . import plink
from . import provenance

# Common arguments/options
verbose = click.option("-v", "--verbose", count=True, help="Increase verbosity")

worker_processes = click.option(
    "-p", "--worker-processes", type=int, default=1, help="Number of worker processes"
)

# TODO help text
chunk_length = click.option(
    "-l",
    "--chunk-length",
    type=int,
    default=None,
    help="Chunk size in the variants dimension",
)

chunk_width = click.option(
    "-w",
    "--chunk-width",
    type=int,
    default=None,
    help="Chunk size in the samples dimension",
)

version = click.version_option(version=provenance.__version__)


# Note: logging hasn't been implemented in the code at all, this is just
# a first pass to try out some ways of doing things to see what works.
def setup_logging(verbosity):
    level = "WARNING"
    if verbosity == 1:
        level = "INFO"
    elif verbosity >= 2:
        level = "DEBUG"
    # NOTE: I'm not that excited about coloredlogs, just trying it out
    # as it is installed by cyvcf2 anyway. We will have some complicated
    # stuff doing on with threads and processes, to logs might not work
    # so well anyway.
    coloredlogs.install(level=level)


@click.command
@click.argument("vcfs", nargs=-1, required=True)
@click.argument("out_path", type=click.Path())
@verbose
@worker_processes
@click.option("-c", "--column-chunk-size", type=int, default=64)
def explode(vcfs, out_path, verbose, worker_processes, column_chunk_size):
    """
    Convert VCF(s) to columnar intermediate format
    """
    setup_logging(verbose)
    vcf.explode(
        vcfs,
        out_path,
        worker_processes=worker_processes,
        column_chunk_size=column_chunk_size,
        show_progress=True,
    )


@click.command
@click.argument("if_path", type=click.Path())
@verbose
def inspect(if_path, verbose):
    """
    Inspect an intermediate format file
    """
    setup_logging(verbose)
    data = vcf.inspect(if_path)
    click.echo(tabulate.tabulate(data, headers="keys"))


@click.command
@click.argument("if_path", type=click.Path())
def mkschema(if_path):
    """
    Generate a schema for zarr encoding
    """
    stream = click.get_text_stream("stdout")
    vcf.mkschema(if_path, stream)


@click.command
@click.argument("if_path", type=click.Path())
@click.argument("zarr_path", type=click.Path())
@verbose
@click.option("-s", "--schema", default=None, type=click.Path(exists=True))
@chunk_length
@chunk_width
@click.option(
    "-V",
    "--max-variant-chunks",
    type=int,
    default=None,
    help=(
        "Truncate the output in the variants dimension to have "
        "this number of chunks. Mainly intended to help with "
        "schema tuning."
    ),
)
@worker_processes
def encode(
    if_path,
    zarr_path,
    verbose,
    schema,
    chunk_length,
    chunk_width,
    max_variant_chunks,
    worker_processes,
):
    """
    Encode intermediate format (see explode) to vcfzarr
    """
    setup_logging(verbose)
    vcf.encode(
        if_path,
        zarr_path,
        schema,
        chunk_length=chunk_length,
        chunk_width=chunk_width,
        max_v_chunks=max_variant_chunks,
        worker_processes=worker_processes,
        show_progress=True,
    )


@click.command(name="convert")
@click.argument("vcfs", nargs=-1, required=True)
@click.argument("out_path", type=click.Path())
@chunk_length
@chunk_width
@verbose
@worker_processes
def convert_vcf(vcfs, out_path, chunk_length, chunk_width, verbose, worker_processes):
    """
    Convert input VCF(s) directly to vcfzarr (not recommended for large files)
    """
    setup_logging(verbose)
    vcf.convert(
        vcfs,
        out_path,
        chunk_length=chunk_length,
        chunk_width=chunk_width,
        show_progress=True,
        worker_processes=worker_processes,
    )


@click.command
@click.argument("vcfs", nargs=-1, required=True)
@click.argument("out_path", type=click.Path())
def validate(vcfs, out_path):
    """
    Development only, do not use. Will be removed before release.
    """
    # FIXME! Will silently not look at remaining VCFs
    vcf.validate(vcfs[0], out_path, show_progress=True)


@version
@click.group()
def vcf2zarr():
    pass


# TODO figure out how to get click to list these in the given order.
vcf2zarr.add_command(explode)
vcf2zarr.add_command(inspect)
vcf2zarr.add_command(mkschema)
vcf2zarr.add_command(encode)
vcf2zarr.add_command(convert_vcf)
vcf2zarr.add_command(validate)


@click.command(name="convert")
@click.argument("in_path", type=click.Path())
@click.argument("out_path", type=click.Path())
@worker_processes
@verbose
@chunk_length
@chunk_width
def convert_plink(
    in_path, out_path, verbose, worker_processes, chunk_length, chunk_width
):
    """
    In development; DO NOT USE!
    """
    setup_logging(verbose)
    plink.convert(
        in_path,
        out_path,
        show_progress=True,
        worker_processes=worker_processes,
        chunk_width=chunk_width,
        chunk_length=chunk_length,
    )


@version
@click.group()
def plink2zarr():
    pass


plink2zarr.add_command(convert_plink)


@click.command
@version
@click.argument("vcf_path", type=click.Path())
@click.option("-i", "--index", type=click.Path(), default=None)
@click.option("-n", "--num-parts", type=int, default=None)
# @click.option("-s", "--part-size", type=int, default=None)
def vcf_partition(vcf_path, index, num_parts):
    indexed_vcf = vcf_utils.IndexedVcf(vcf_path, index)
    regions = indexed_vcf.partition_into_regions(num_parts=num_parts)
    click.echo("\n".join(map(str, regions)))
