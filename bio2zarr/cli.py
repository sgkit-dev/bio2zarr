import click
import tabulate
import coloredlogs

from . import vcf
from . import vcf_utils
from . import plink
from . import provenance

class NaturalOrderGroup(click.Group):
    """
    List commands in the order they are provided in the help text.
    """

    def list_commands(self, ctx):
        return self.commands.keys()


# Common arguments/options
verbose = click.option("-v", "--verbose", count=True, help="Increase verbosity")

version = click.version_option(version=f"{provenance.__version__}")

worker_processes = click.option(
    "-p", "--worker-processes", type=int, default=1, help="Number of worker processes"
)

column_chunk_size = click.option(
    "-c",
    "--column-chunk-size",
    type=int,
    default=64,
    help="Approximate uncompressed size of exploded column chunks in MiB",
)

# Note: -l and -w were chosen when these were called "width" and "length".
# possibly there are better letters now.
variants_chunk_size = click.option(
    "-l",
    "--variants-chunk-size",
    type=int,
    default=None,
    help="Chunk size in the variants dimension",
)

samples_chunk_size = click.option(
    "-w",
    "--samples-chunk-size",
    type=int,
    default=None,
    help="Chunk size in the samples dimension",
)


def setup_logging(verbosity):
    level = "WARNING"
    if verbosity == 1:
        level = "INFO"
    elif verbosity >= 2:
        level = "DEBUG"
    # NOTE: I'm not that excited about coloredlogs, just trying it out
    # as it is installed by cyvcf2 anyway.
    coloredlogs.install(level=level)


@click.command
@click.argument("vcfs", nargs=-1, required=True)
@click.argument("out_path", type=click.Path())
@verbose
@worker_processes
@column_chunk_size
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
@click.argument("vcfs", nargs=-1, required=True)
@click.argument("out_path", type=click.Path())
@click.option("-n", "--target-num-partitions", type=int, required=True)
@verbose
@worker_processes
def dexplode_init(vcfs, out_path, target_num_partitions, verbose, worker_processes):
    """
    Initial step for parallel conversion of VCF(s) to columnar intermediate format
    """
    setup_logging(verbose)
    vcf.explode_init(
        vcfs,
        out_path,
        target_num_partitions=target_num_partitions,
        worker_processes=worker_processes,
        show_progress=True,
    )


@click.command
@click.argument("path", type=click.Path())
def dexplode_partition_count(path):
    """
    Count the actual number of partitions in a parallel conversion of VCF(s) to columnar intermediate format
    """
    click.echo(vcf.explode_partition_count(path))


@click.command
@click.argument("path", type=click.Path(), required=True)
@click.option("-s", "--start", type=int, required=True)
@click.option("-e", "--end", type=int, required=True)
@verbose
@worker_processes
@column_chunk_size
def dexplode_slice(path, start, end, verbose, worker_processes, column_chunk_size):
    """
    Convert VCF(s) to columnar intermediate format
    """
    setup_logging(verbose)
    vcf.explode_slice(
        path,
        start,
        end,
        worker_processes=worker_processes,
        column_chunk_size=column_chunk_size,
        show_progress=True,
    )


@click.command
@click.argument("path", type=click.Path(), required=True)
@verbose
def dexplode_finalise(path, verbose):
    """
    Final step for parallel conversion of VCF(s) to columnar intermediate format
    """
    setup_logging(verbose)
    vcf.explode_finalise(path)


@click.command
@click.argument("if_path", type=click.Path())
@verbose
def inspect(if_path, verbose):
    """
    Inspect an intermediate format or Zarr path.
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
@variants_chunk_size
@samples_chunk_size
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
@click.option(
    "-M",
    "--max-memory",
    type=int,
    default=None,
    help="An approximate bound on overall memory usage in megabytes",
)
@worker_processes
def encode(
    if_path,
    zarr_path,
    verbose,
    schema,
    variants_chunk_size,
    samples_chunk_size,
    max_variant_chunks,
    max_memory,
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
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
        max_v_chunks=max_variant_chunks,
        worker_processes=worker_processes,
        max_memory=max_memory,
        show_progress=True,
    )


@click.command(name="convert")
@click.argument("vcfs", nargs=-1, required=True)
@click.argument("out_path", type=click.Path())
@variants_chunk_size
@samples_chunk_size
@verbose
@worker_processes
def convert_vcf(
    vcfs, out_path, variants_chunk_size, samples_chunk_size, verbose, worker_processes
):
    """
    Convert input VCF(s) directly to vcfzarr (not recommended for large files)
    """
    setup_logging(verbose)
    vcf.convert(
        vcfs,
        out_path,
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
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
@click.group(cls=NaturalOrderGroup)
def vcf2zarr():
    pass


# TODO figure out how to get click to list these in the given order.
vcf2zarr.add_command(convert_vcf)
vcf2zarr.add_command(explode)
vcf2zarr.add_command(inspect)
vcf2zarr.add_command(mkschema)
vcf2zarr.add_command(encode)
vcf2zarr.add_command(dexplode_init)
vcf2zarr.add_command(dexplode_partition_count)
vcf2zarr.add_command(dexplode_slice)
vcf2zarr.add_command(dexplode_finalise)
vcf2zarr.add_command(validate)


@click.command(name="convert")
@click.argument("in_path", type=click.Path())
@click.argument("out_path", type=click.Path())
@worker_processes
@verbose
@variants_chunk_size
@samples_chunk_size
def convert_plink(
    in_path,
    out_path,
    verbose,
    worker_processes,
    variants_chunk_size,
    samples_chunk_size,
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
        samples_chunk_size=samples_chunk_size,
        variants_chunk_size=variants_chunk_size,
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
