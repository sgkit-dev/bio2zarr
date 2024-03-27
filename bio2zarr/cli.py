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
@click.argument("zarr_path", type=click.Path())
@verbose
@worker_processes
@column_chunk_size
def explode(vcfs, zarr_path, verbose, worker_processes, column_chunk_size):
    """
    Convert VCF(s) to intermediate columnar format
    """
    setup_logging(verbose)
    vcf.explode(
        vcfs,
        zarr_path,
        worker_processes=worker_processes,
        column_chunk_size=column_chunk_size,
        show_progress=True,
    )


@click.command
@click.argument("vcfs", nargs=-1, required=True)
@click.argument("icf_path", type=click.Path())
@click.argument("num_partitions", type=int)
@column_chunk_size
@verbose
@worker_processes
def dexplode_init(
    vcfs, icf_path, num_partitions, column_chunk_size, verbose, worker_processes
):
    """
    Initial step for parallel conversion of VCF(s) to intermediate columnar format
    over the requested number of paritions.
    """
    setup_logging(verbose)
    num_partitions = vcf.explode_init(
        icf_path,
        vcfs,
        target_num_partitions=num_partitions,
        column_chunk_size=column_chunk_size,
        worker_processes=worker_processes,
        show_progress=True,
    )
    click.echo(num_partitions)


@click.command
@click.argument("icf_path", type=click.Path())
@click.argument("partition", type=int)
@verbose
def dexplode_partition(icf_path, partition, verbose):
    """
    Convert a VCF partition into intermediate columnar format. Must be called *after*
    the ICF path has been initialised with dexplode_init. Partition indexes must be
    from 0 (inclusive) to the number of paritions returned by dexplode_init (exclusive).
    """
    setup_logging(verbose)
    vcf.explode_partition(icf_path, partition, show_progress=True)


@click.command
@click.argument("path", type=click.Path(), required=True)
@verbose
def dexplode_finalise(path, verbose):
    """
    Final step for parallel conversion of VCF(s) to intermediate columnar format
    """
    setup_logging(verbose)
    vcf.explode_finalise(path)


@click.command
@click.argument("icf_path", type=click.Path())
@verbose
def inspect(icf_path, verbose):
    """
    Inspect an intermediate format or Zarr path.
    """
    setup_logging(verbose)
    data = vcf.inspect(icf_path)
    click.echo(tabulate.tabulate(data, headers="keys"))


@click.command
@click.argument("icf_path", type=click.Path())
def mkschema(icf_path):
    """
    Generate a schema for zarr encoding
    """
    stream = click.get_text_stream("stdout")
    vcf.mkschema(icf_path, stream)


@click.command
@click.argument("icf_path", type=click.Path())
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
    icf_path,
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
    Encode intermediate columnar format (see explode) to vcfzarr.
    """
    setup_logging(verbose)
    vcf.encode(
        icf_path,
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
@click.argument("zarr_path", type=click.Path())
@variants_chunk_size
@samples_chunk_size
@verbose
@worker_processes
def convert_vcf(
    vcfs, zarr_path, variants_chunk_size, samples_chunk_size, verbose, worker_processes
):
    """
    Convert input VCF(s) directly to vcfzarr (not recommended for large files).
    """
    setup_logging(verbose)
    vcf.convert(
        vcfs,
        zarr_path,
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
        show_progress=True,
        worker_processes=worker_processes,
    )


@click.command
@click.argument("vcfs", nargs=-1, required=True)
@click.argument("zarr_path", type=click.Path())
def validate(vcfs, zarr_path):
    """
    Development only, do not use. Will be removed before release.
    """
    # FIXME! Will silently not look at remaining VCFs
    vcf.validate(vcfs[0], zarr_path, show_progress=True)


@version
@click.group(cls=NaturalOrderGroup)
def vcf2zarr():
    """
    Convert VCF file(s) to the vcfzarr format.

    The simplest usage is:

    $ vcf2zarr convert [VCF_FILE] [ZARR_PATH]

    This will convert the indexed VCF (or BCF) into the vcfzarr format in a single
    step. As this writes the intermediate columnar format to a temporary directory,
    we only recommend this approach for small files (< 1GB, say).

    The recommended approach is to run the conversion in two passes, and
    to keep the intermediate columnar format ("exploded") around to facilitate
    experimentation with chunk sizes and compression settings:

    \b
    $ vcf2zarr explode [VCF_FILE_1] ... [VCF_FILE_N] [ICF_PATH]
    $ vcf2zarr encode [ICF_PATH] [ZARR_PATH]

    The inspect command provides a way to view contents of an exploded ICF
    or Zarr:

    $ vcf2zarr inspect [PATH]

    This is useful when tweaking chunk sizes and compression settings to suit
    your dataset, using the mkschema command and --schema option to encode:

    \b
    $ vcf2zarr mkschema [ICF_PATH] > schema.json
    $ vcf2zarr encode [ICF_PATH] [ZARR_PATH] --schema schema.json

    By editing the schema.json file you can drop columns that are not of interest
    and edit column specific compression settings. The --max-variant-chunks option
    to encode allows you to try out these options on small subsets, hopefully
    arriving at settings with the desired balance of compression and query
    performance.

    ADVANCED USAGE

    For very large datasets (terabyte scale) it may be necessary to distribute the
    explode and encode steps across a cluster:

    \b
    $ vcf2zarr dexplode-init [VCF_FILE_1] ... [VCF_FILE_N] [ICF_PATH] [NUM_PARTITIONS]
    $ vcf2zarr dexplode-partition [ICF_PATH] [PARTITION_INDEX]
    $ vcf2zarr dexplode-finalise [ICF_PATH]

    See the online documentation at [FIXME] for more details on distributed explode.
    """


# TODO figure out how to get click to list these in the given order.
vcf2zarr.add_command(convert_vcf)
vcf2zarr.add_command(inspect)
vcf2zarr.add_command(explode)
vcf2zarr.add_command(mkschema)
vcf2zarr.add_command(encode)
vcf2zarr.add_command(dexplode_init)
vcf2zarr.add_command(dexplode_partition)
vcf2zarr.add_command(dexplode_finalise)
vcf2zarr.add_command(validate)


@click.command(name="convert")
@click.argument("in_path", type=click.Path())
@click.argument("zarr_path", type=click.Path())
@worker_processes
@verbose
@variants_chunk_size
@samples_chunk_size
def convert_plink(
    in_path,
    zarr_path,
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
        zarr_path,
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
