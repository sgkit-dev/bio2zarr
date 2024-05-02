import logging
import os
import pathlib
import shutil

import click
import coloredlogs
import humanfriendly
import numcodecs
import tabulate

from . import plink, provenance, vcf, vcf_utils

logger = logging.getLogger(__name__)


class NaturalOrderGroup(click.Group):
    """
    List commands in the order they are provided in the help text.
    """

    def list_commands(self, ctx):
        return self.commands.keys()


# Common arguments/options
vcfs = click.argument(
    "vcfs", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False)
)

new_icf_path = click.argument(
    "icf_path", type=click.Path(file_okay=False, dir_okay=True)
)

icf_path = click.argument(
    "icf_path", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)

new_zarr_path = click.argument(
    "zarr_path", type=click.Path(file_okay=False, dir_okay=True)
)

zarr_path = click.argument(
    "zarr_path", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)

num_partitions = click.argument("num_partitions", type=click.IntRange(min=1))

partition = click.argument("partition", type=click.IntRange(min=0))

verbose = click.option("-v", "--verbose", count=True, help="Increase verbosity")

force = click.option(
    "-f",
    "--force",
    is_flag=True,
    flag_value=True,
    help="Force overwriting of existing directories",
)

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

# We could provide the full flexiblity of numcodecs/Blosc here, but there
# doesn't seem much point. Can always add more arguments here to control
# compression level, etc.
compressor = click.option(
    "-C",
    "--compressor",
    type=click.Choice(["lz4", "zstd"]),
    default=None,
    help="Codec to use for compressing column chunks (Default=zstd).",
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

schema = click.option("-s", "--schema", default=None, type=click.Path(exists=True))

max_variant_chunks = click.option(
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

max_memory = click.option(
    "-M",
    "--max-memory",
    default=None,
    help="An approximate bound on overall memory usage (e.g. 10G),",
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


def check_overwrite_dir(path, force):
    path = pathlib.Path(path)
    if path.exists():
        if not force:
            click.confirm(
                f"Do you want to overwrite {path}? (use --force to skip this check)",
                abort=True,
            )
        # These trees can be mondo-big and on slow file systems, so it's entirely
        # feasible that the delete would fail or be killed. This makes it less likely
        # that partially deleted paths are mistaken for good paths.
        tmp_delete_path = path.with_suffix(f"{path.suffix}.{os.getpid()}.DELETING")
        logger.info(f"Deleting {path} (renamed to {tmp_delete_path} while in progress)")
        os.rename(path, tmp_delete_path)
        shutil.rmtree(tmp_delete_path)


def get_compressor(cname):
    if cname is None:
        return None
    config = vcf.ICF_DEFAULT_COMPRESSOR.get_config()
    config["cname"] = cname
    return numcodecs.get_codec(config)


@click.command
@vcfs
@new_icf_path
@force
@verbose
@column_chunk_size
@compressor
@worker_processes
def explode(
    vcfs, icf_path, force, verbose, column_chunk_size, compressor, worker_processes
):
    """
    Convert VCF(s) to intermediate columnar format
    """
    setup_logging(verbose)
    check_overwrite_dir(icf_path, force)
    vcf.explode(
        icf_path,
        vcfs,
        worker_processes=worker_processes,
        column_chunk_size=column_chunk_size,
        compressor=get_compressor(compressor),
        show_progress=True,
    )


@click.command
@vcfs
@new_icf_path
@num_partitions
@force
@column_chunk_size
@compressor
@verbose
@worker_processes
def dexplode_init(
    vcfs,
    icf_path,
    num_partitions,
    force,
    column_chunk_size,
    compressor,
    verbose,
    worker_processes,
):
    """
    Initial step for distributed conversion of VCF(s) to intermediate columnar format
    over the requested number of paritions.
    """
    setup_logging(verbose)
    check_overwrite_dir(icf_path, force)
    num_partitions = vcf.explode_init(
        icf_path,
        vcfs,
        target_num_partitions=num_partitions,
        column_chunk_size=column_chunk_size,
        worker_processes=worker_processes,
        compressor=get_compressor(compressor),
        show_progress=True,
    )
    click.echo(num_partitions)


@click.command
@icf_path
@partition
@verbose
def dexplode_partition(icf_path, partition, verbose):
    """
    Convert a VCF partition to intermediate columnar format. Must be called *after*
    the ICF path has been initialised with dexplode_init. Partition indexes must be
    from 0 (inclusive) to the number of paritions returned by dexplode_init (exclusive).
    """
    setup_logging(verbose)
    vcf.explode_partition(icf_path, partition)


@click.command
@icf_path
@verbose
def dexplode_finalise(icf_path, verbose):
    """
    Final step for distributed conversion of VCF(s) to intermediate columnar format.
    """
    setup_logging(verbose)
    vcf.explode_finalise(icf_path)


@click.command
@click.argument("path", type=click.Path())
@verbose
def inspect(path, verbose):
    """
    Inspect an intermediate columnar format or Zarr path.
    """
    setup_logging(verbose)
    data = vcf.inspect(path)
    click.echo(tabulate.tabulate(data, headers="keys"))


@click.command
@icf_path
def mkschema(icf_path):
    """
    Generate a schema for zarr encoding
    """
    stream = click.get_text_stream("stdout")
    vcf.mkschema(icf_path, stream)


@click.command
@icf_path
@new_zarr_path
@force
@verbose
@schema
@variants_chunk_size
@samples_chunk_size
@max_variant_chunks
@max_memory
@worker_processes
def encode(
    icf_path,
    zarr_path,
    force,
    verbose,
    schema,
    variants_chunk_size,
    samples_chunk_size,
    max_variant_chunks,
    max_memory,
    worker_processes,
):
    """
    Convert intermediate columnar format to vcfzarr.
    """
    setup_logging(verbose)
    check_overwrite_dir(zarr_path, force)
    vcf.encode(
        icf_path,
        zarr_path,
        schema_path=schema,
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
        max_variant_chunks=max_variant_chunks,
        worker_processes=worker_processes,
        max_memory=max_memory,
        show_progress=True,
    )


@click.command
@icf_path
@new_zarr_path
@num_partitions
@force
@schema
@variants_chunk_size
@samples_chunk_size
@max_variant_chunks
@verbose
def dencode_init(
    icf_path,
    zarr_path,
    num_partitions,
    force,
    schema,
    variants_chunk_size,
    samples_chunk_size,
    max_variant_chunks,
    verbose,
):
    """
    Initialise conversion of intermediate format to VCF Zarr. This will
    set up the specified ZARR_PATH to perform this conversion over
    NUM_PARTITIONS.

    The output of this commmand is the actual number of partitions generated
    (which may be less then the requested number, if there is not sufficient
    chunks in the variants dimension) and a rough lower-bound on the amount
    of memory required to encode a partition.

    NOTE: the format of this output will likely change in subsequent releases;
    it should not be considered machine-readable for now.
    """
    setup_logging(verbose)
    check_overwrite_dir(zarr_path, force)
    num_partitions, max_memory = vcf.encode_init(
        icf_path,
        zarr_path,
        target_num_partitions=num_partitions,
        schema_path=schema,
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
        max_variant_chunks=max_variant_chunks,
        show_progress=True,
    )
    formatted_size = humanfriendly.format_size(max_memory, binary=True)
    # NOTE adding the size to the stdout here so that users can parse it
    # and use in their submission scripts. This is a first pass, and
    # will most likely change as we see what works and doesn't.
    # NOTE we probably want to format this as a table, which lists
    # some other properties, line by line
    # NOTE This size number is also not quite enough, you need a bit of
    # headroom with it (probably 10% or so). We should include this.
    click.echo(f"{num_partitions}\t{formatted_size}")


@click.command
@zarr_path
@partition
@verbose
def dencode_partition(zarr_path, partition, verbose):
    """
    Convert a partition from intermediate columnar format to VCF Zarr.
    Must be called *after* the Zarr path has been initialised with dencode_init.
    Partition indexes must be from 0 (inclusive) to the number of paritions
    returned by dencode_init (exclusive).
    """
    setup_logging(verbose)
    vcf.encode_partition(zarr_path, partition)


@click.command
@zarr_path
@verbose
def dencode_finalise(zarr_path, verbose):
    """
    Final step for distributed conversion of ICF to VCF Zarr.
    """
    setup_logging(verbose)
    vcf.encode_finalise(zarr_path, show_progress=True)


@click.command(name="convert")
@vcfs
@new_zarr_path
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
vcf2zarr.add_command(dencode_init)
vcf2zarr.add_command(dencode_partition)
vcf2zarr.add_command(dencode_finalise)


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
