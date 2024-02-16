import click
import tabulate

# import bio2zarr.vcf as cnv  # fixme
from . import vcf as cnv


@click.command
@click.argument("vcfs", nargs=-1, required=True)
@click.argument("out_path", type=click.Path())
@click.option("-p", "--worker-processes", type=int, default=1)
@click.option("-c", "--column-chunk-size", type=int, default=64)
def explode(vcfs, out_path, worker_processes, column_chunk_size):
    cnv.explode(
        vcfs,
        out_path,
        worker_processes=worker_processes,
        column_chunk_size=column_chunk_size,
        show_progress=True,
    )


@click.command
@click.argument("columnarised", type=click.Path())
def summarise(columnarised):
    pcvcf = cnv.PickleChunkedVcf.load(columnarised)
    data = pcvcf.summary_table()
    print(tabulate.tabulate(data, headers="keys"))


@click.command
@click.argument("columnarised", type=click.Path())
# @click.argument("specfile", type=click.Path())
def genspec(columnarised):
    stream = click.get_text_stream("stdout")
    cnv.generate_spec(columnarised, stream)


@click.command
@click.argument("columnarised", type=click.Path())
@click.argument("zarr_path", type=click.Path())
@click.option("-s", "--conversion-spec", default=None)
@click.option("-p", "--worker-processes", type=int, default=1)
def to_zarr(columnarised, zarr_path, conversion_spec, worker_processes):
    cnv.to_zarr(
        columnarised,
        zarr_path,
        conversion_spec,
        worker_processes=worker_processes,
        show_progress=True,
    )


@click.command(name="convert")
@click.argument("vcfs", nargs=-1, required=True)
@click.argument("out_path", type=click.Path())
@click.option("-p", "--worker-processes", type=int, default=1)
def convert_vcf(vcfs, out_path, worker_processes):
    cnv.convert_vcf(
        vcfs, out_path, show_progress=True, worker_processes=worker_processes
    )


@click.command
@click.argument("vcfs", nargs=-1, required=True)
@click.argument("out_path", type=click.Path())
def validate(vcfs, out_path):
    cnv.validate(vcfs[0], out_path, show_progress=True)


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
@click.option("-p", "--worker-processes", type=int, default=1)
@click.option("--chunk-width", type=int, default=None)
@click.option("--chunk-length", type=int, default=None)
def convert_plink(plink, out_path, worker_processes, chunk_width, chunk_length):
    cnv.convert_plink(
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
