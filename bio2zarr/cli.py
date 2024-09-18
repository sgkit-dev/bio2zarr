import logging
import os
import pathlib
import shutil

import click
import coloredlogs
import numcodecs
import tabulate

from . import bed2zarr, plink, provenance, vcf2zarr, vcf_utils
from .vcf2zarr import icf as icf_mod

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

num_partitions = click.option(
    "-n",
    "--num-partitions",
    type=click.IntRange(min=1),
    default=None,
    help="Target number of partitions to split into",
)

partition = click.argument("partition", type=click.IntRange(min=0))

verbose = click.option("-v", "--verbose", count=True, help="Increase verbosity")

force = click.option(
    "-f",
    "--force",
    is_flag=True,
    flag_value=True,
    help="Force overwriting of existing directories",
)

progress = click.option(
    "-P /-Q",
    "--progress/--no-progress",
    default=True,
    help="Show progress bars (default: show)",
)

one_based = click.option(
    "--one-based",
    is_flag=True,
    flag_value=True,
    help="Partition indexes are interpreted as one-based",
)

json = click.option(
    "--json",
    is_flag=True,
    flag_value=True,
    help="Output summary data in JSON format",
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

local_alleles = click.option(
    "--local-alleles/--no-local-alleles",
    show_default=True,
    default=False,
    help="Use local allele fields to reduce the storage requirements of the output.",
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


def check_partitions(num_partitions):
    if num_partitions is None:
        raise click.UsageError(
            "-n/--num-partitions must currently be specified. Future versions "
            "will provide reasonable defaults or other means of specifying "
            "partitions."
        )


def get_compressor(cname):
    if cname is None:
        return None
    config = icf_mod.ICF_DEFAULT_COMPRESSOR.get_config()
    config["cname"] = cname
    return numcodecs.get_codec(config)


def show_work_summary(work_summary, json):
    if json:
        output = work_summary.asjson()
    else:
        data = work_summary.asdict()
        output = tabulate.tabulate(list(data.items()), tablefmt="plain")
        # output = "\n".join(f"{k}\t{v}" for k, v in data.items())
    click.echo(output)


@click.command
@vcfs
@new_icf_path
@force
@verbose
@column_chunk_size
@compressor
@progress
@worker_processes
@local_alleles
def explode(
    vcfs,
    icf_path,
    force,
    verbose,
    column_chunk_size,
    compressor,
    progress,
    worker_processes,
    local_alleles,
):
    """
    Convert VCF(s) to intermediate columnar format
    """
    setup_logging(verbose)
    check_overwrite_dir(icf_path, force)
    vcf2zarr.explode(
        icf_path,
        vcfs,
        worker_processes=worker_processes,
        column_chunk_size=column_chunk_size,
        compressor=get_compressor(compressor),
        show_progress=progress,
        local_alleles=local_alleles,
    )


@click.command
@vcfs
@new_icf_path
@num_partitions
@force
@column_chunk_size
@compressor
@json
@verbose
@progress
@worker_processes
@local_alleles
def dexplode_init(
    vcfs,
    icf_path,
    num_partitions,
    force,
    column_chunk_size,
    compressor,
    json,
    verbose,
    progress,
    worker_processes,
    local_alleles,
):
    """
    Initial step for distributed conversion of VCF(s) to intermediate columnar format
    over some number of paritions.
    """
    setup_logging(verbose)
    check_overwrite_dir(icf_path, force)
    check_partitions(num_partitions)
    work_summary = vcf2zarr.explode_init(
        icf_path,
        vcfs,
        target_num_partitions=num_partitions,
        column_chunk_size=column_chunk_size,
        worker_processes=worker_processes,
        compressor=get_compressor(compressor),
        show_progress=progress,
        local_alleles=local_alleles,
    )
    show_work_summary(work_summary, json)


@click.command
@icf_path
@partition
@verbose
@one_based
def dexplode_partition(icf_path, partition, verbose, one_based):
    """
    Convert a VCF partition to intermediate columnar format. Must be called
    after the ICF path has been initialised with dexplode_init. By default,
    partition indexes are from 0 to the number of partitions N (returned by
    dexplode_init), exclusive. If the --one-based option is specifed,
    partition indexes are in the range 1 to N, inclusive.
    """
    setup_logging(verbose)
    if one_based:
        partition -= 1
    vcf2zarr.explode_partition(icf_path, partition)


@click.command
@icf_path
@verbose
def dexplode_finalise(icf_path, verbose):
    """
    Final step for distributed conversion of VCF(s) to intermediate columnar format.
    """
    setup_logging(verbose)
    vcf2zarr.explode_finalise(icf_path)


@click.command
@click.argument("path", type=click.Path())
@verbose
def inspect(path, verbose):
    """
    Inspect an intermediate columnar format or Zarr path.
    """
    setup_logging(verbose)
    data = vcf2zarr.inspect(path)
    click.echo(tabulate.tabulate(data, headers="keys"))


@click.command
@icf_path
def mkschema(icf_path):
    """
    Generate a schema for zarr encoding
    """
    stream = click.get_text_stream("stdout")
    vcf2zarr.mkschema(icf_path, stream)


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
@progress
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
    progress,
    worker_processes,
):
    """
    Convert intermediate columnar format to vcfzarr.
    """
    setup_logging(verbose)
    check_overwrite_dir(zarr_path, force)
    vcf2zarr.encode(
        icf_path,
        zarr_path,
        schema_path=schema,
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
        max_variant_chunks=max_variant_chunks,
        worker_processes=worker_processes,
        max_memory=max_memory,
        show_progress=progress,
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
@json
@progress
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
    json,
    progress,
    verbose,
):
    """
    Initialise conversion of intermediate format to VCF Zarr. This will
    set up the specified ZARR_PATH to perform this conversion over
    some number of partitions.

    The output of this commmand is the actual number of partitions generated
    (which may be less then the requested number, if there is not sufficient
    chunks in the variants dimension) and a rough lower-bound on the amount
    of memory required to encode a partition.

    NOTE: the format of this output will likely change in subsequent releases;
    it should not be considered machine-readable for now.
    """
    setup_logging(verbose)
    check_overwrite_dir(zarr_path, force)
    check_partitions(num_partitions)
    work_summary = vcf2zarr.encode_init(
        icf_path,
        zarr_path,
        target_num_partitions=num_partitions,
        schema_path=schema,
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
        max_variant_chunks=max_variant_chunks,
        show_progress=progress,
    )
    show_work_summary(work_summary, json)


@click.command
@zarr_path
@partition
@verbose
@one_based
def dencode_partition(zarr_path, partition, verbose, one_based):
    """
    Convert a partition from intermediate columnar format to VCF Zarr. Must be
    called after the Zarr path has been initialised with dencode_init. By
    default, partition indexes are from 0 to the number of partitions N
    (returned by dencode_init), exclusive. If the --one-based option is
    specifed, partition indexes are in the range 1 to N, inclusive."""
    setup_logging(verbose)
    if one_based:
        partition -= 1
    vcf2zarr.encode_partition(zarr_path, partition)


@click.command
@zarr_path
@verbose
@progress
def dencode_finalise(zarr_path, verbose, progress):
    """
    Final step for distributed conversion of ICF to VCF Zarr.
    """
    setup_logging(verbose)
    vcf2zarr.encode_finalise(zarr_path, show_progress=progress)


@click.command(name="convert")
@vcfs
@new_zarr_path
@force
@variants_chunk_size
@samples_chunk_size
@verbose
@progress
@worker_processes
@local_alleles
def convert_vcf(
    vcfs,
    zarr_path,
    force,
    variants_chunk_size,
    samples_chunk_size,
    verbose,
    progress,
    worker_processes,
    local_alleles,
):
    """
    Convert input VCF(s) directly to vcfzarr (not recommended for large files).
    """
    setup_logging(verbose)
    check_overwrite_dir(zarr_path, force)
    vcf2zarr.convert(
        vcfs,
        zarr_path,
        variants_chunk_size=variants_chunk_size,
        samples_chunk_size=samples_chunk_size,
        show_progress=progress,
        worker_processes=worker_processes,
        local_alleles=local_alleles,
    )


@version
@click.group(cls=NaturalOrderGroup, name="vcf2zarr")
def vcf2zarr_main():
    """
    Convert VCF file(s) to the vcfzarr format.

    See the online documentation at https://sgkit-dev.github.io/bio2zarr/
    for more information.
    """


vcf2zarr_main.add_command(convert_vcf)
vcf2zarr_main.add_command(inspect)
vcf2zarr_main.add_command(explode)
vcf2zarr_main.add_command(mkschema)
vcf2zarr_main.add_command(encode)
vcf2zarr_main.add_command(dexplode_init)
vcf2zarr_main.add_command(dexplode_partition)
vcf2zarr_main.add_command(dexplode_finalise)
vcf2zarr_main.add_command(dencode_init)
vcf2zarr_main.add_command(dencode_partition)
vcf2zarr_main.add_command(dencode_finalise)


@click.command(name="convert")
@click.argument("in_path", type=click.Path())
@click.argument("zarr_path", type=click.Path())
@worker_processes
@progress
@verbose
@variants_chunk_size
@samples_chunk_size
def convert_plink(
    in_path,
    zarr_path,
    verbose,
    worker_processes,
    progress,
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
        show_progress=progress,
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
@click.argument(
    "bed_path",
    type=click.Path(exists=True, dir_okay=False),
)
@new_zarr_path
@verbose
@force
@progress
def bed2zarr_main(bed_path, zarr_path, verbose, force, progress):
    """
    Convert BED file to the Zarr format. The BED regions will be
    converted to binary-encoded arrays whose length is equal to the
    length of the reference genome. The BED file regions are used to
    mask the reference genome, where the masked regions are set to 1
    and the unmasked regions are set to 0.

    The BED file must be compressed and tabix-indexed.
    """
    setup_logging(verbose)
    check_overwrite_dir(zarr_path, force)
    bed2zarr.bed2zarr(
        bed_path,
        zarr_path,
        show_progress=progress,
    )


@click.command
@version
@vcfs
@verbose
@num_partitions
@click.option(
    "-s",
    "--partition-size",
    type=str,
    default=None,
    help="Target (compressed) size of VCF partitions, e.g. 100KB, 10MiB, 1G.",
)
def vcfpartition(vcfs, verbose, num_partitions, partition_size):
    """
    Output bcftools region strings that partition the indexed VCF/BCF files
    into either an approximate number of parts (-n), or parts of approximately
    a given size (-s). One of -n or -s must be supplied.

    If multiple VCF/BCF files are provided, the number of parts (-n) is
    interpreted as the total number of partitions across all the files,
    and the partitions are distributed evenly among the files.

    Note that both the number of partitions and sizes are a target, and the
    returned number of partitions may not exactly correspond. In particular,
    there is a maximum level of granularity determined by the associated index
    which cannot be exceeded.

    Note also that the partitions returned may vary considerably in the number
    of records that they contain.
    """
    setup_logging(verbose)
    if num_partitions is None and partition_size is None:
        raise click.UsageError(
            "Either --num-partitions or --partition-size must be specified"
        )

    if num_partitions is None:
        num_parts_per_path = None
    else:
        num_parts_per_path = max(1, num_partitions // len(vcfs))

    for vcf_path in vcfs:
        indexed_vcf = vcf_utils.IndexedVcf(vcf_path)
        regions = indexed_vcf.partition_into_regions(
            num_parts=num_parts_per_path, target_part_size=partition_size
        )
        for region in regions:
            click.echo(f"{region}\t{vcf_path}")
