import click
import tabulate
import coloredlogs

from . import vcf

# Common arguments/options
verbose = click.option("-v", "--verbose", count=True, help="Increase verbosity")

worker_processes = click.option(
    "-p", "--worker-processes", type=int, default=1, help="Number of worker processes"
)


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
    setup_logging(verbose)
    vcf.explode(
        vcfs,
        out_path,
        worker_processes=worker_processes,
        column_chunk_size=column_chunk_size,
        show_progress=True,
    )


@click.command
@click.argument("columnarised", type=click.Path())
@verbose
def summarise(columnarised, verbose):
    setup_logging(verbose)
    pcvcf = vcf.PickleChunkedVcf.load(columnarised)
    data = pcvcf.summary_table()
    click.echo(tabulate.tabulate(data, headers="keys"))


@click.command
@click.argument("columnarised", type=click.Path())
# @click.argument("specfile", type=click.Path())
def genspec(columnarised):
    stream = click.get_text_stream("stdout")
    vcf.generate_spec(columnarised, stream)


@click.command
@click.argument("columnarised", type=click.Path())
@click.argument("zarr_path", type=click.Path())
@verbose
@click.option("-s", "--conversion-spec", default=None)
@worker_processes
def to_zarr(columnarised, zarr_path, verbose, conversion_spec, worker_processes):
    setup_logging(verbose)
    vcf.to_zarr(
        columnarised,
        zarr_path,
        conversion_spec,
        worker_processes=worker_processes,
        show_progress=True,
    )


@click.command(name="convert")
@click.argument("vcfs", nargs=-1, required=True)
@click.argument("out_path", type=click.Path())
@verbose
@worker_processes
def convert_vcf(vcfs, out_path, verbose, worker_processes):
    setup_logging(verbose)
    vcf.convert_vcf(
        vcfs, out_path, show_progress=True, worker_processes=worker_processes
    )


@click.command
@click.argument("vcfs", nargs=-1, required=True)
@click.argument("out_path", type=click.Path())
def validate(vcfs, out_path):
    vcf.validate(vcfs[0], out_path, show_progress=True)


@click.group()
def vcf2zarr():
    pass


vcf2zarr.add_command(explode)
vcf2zarr.add_command(summarise)
vcf2zarr.add_command(genspec)
vcf2zarr.add_command(to_zarr)
vcf2zarr.add_command(convert_vcf)
vcf2zarr.add_command(validate)


@click.command(name="convert")
@click.argument("plink", type=click.Path())
@click.argument("out_path", type=click.Path())
@worker_processes
@click.option("--chunk-width", type=int, default=None)
@click.option("--chunk-length", type=int, default=None)
def convert_plink(plink, out_path, worker_processes, chunk_width, chunk_length):
    vcf.convert_plink(
        plink,
        out_path,
        show_progress=True,
        worker_processes=worker_processes,
        chunk_width=chunk_width,
        chunk_length=chunk_length,
    )


@click.group()
def plink2zarr():
    pass


plink2zarr.add_command(convert_plink)
