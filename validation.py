# Development script to automate running the validation tests.
# These are large-scale tests that are not possible to run
# under unit-test conditions.
import pathlib
import click


from bio2zarr import vcf


@click.command
@click.argument("vcfs", nargs=-1)
@click.option("-p", "--worker-processes", type=int, default=1)
@click.option("-f", "--force", is_flag=True, default=False)
def cli(vcfs, worker_processes, force):
    data_path = pathlib.Path("validation-data")
    if len(vcfs) == 0:
        vcfs = list(data_path.glob("*.vcf.gz")) + list(data_path.glob("*.bcf"))
    else:
        vcfs = [pathlib.Path(f) for f in vcfs]
    tmp_path = pathlib.Path("validation-tmp")
    tmp_path.mkdir(exist_ok=True)
    for f in vcfs:
        print(f)
        exploded = tmp_path / (f.name + ".exploded")
        if force or not exploded.exists():
            vcf.explode(
                [f],
                exploded,
                worker_processes=worker_processes,
                show_progress=True,
            )
        spec = tmp_path / (f.name + ".schema")
        if force or not spec.exists():
            with open(spec, "w") as specfile:
                vcf.mkschema(exploded, specfile)

        zarr = tmp_path / (f.name + ".zarr")
        if force or not zarr.exists():
            vcf.encode(
                exploded,
                zarr,
                spec,
                worker_processes=worker_processes,
                show_progress=True,
            )

        vcf.validate(f, zarr, show_progress=True)


if __name__ == "__main__":
    cli()
