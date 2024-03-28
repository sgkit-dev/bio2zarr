# Development script to automate running the validation tests.
# These are large-scale tests that are not possible to run
# under unit-test conditions.
import pathlib
import shutil
import click


from bio2zarr import vcf

# TODO add support here for split vcfs. Perhaps simplest to take a
# directory provided as input as indicating this, and then having
# the original unsplit vs split files in there following some
# naming conventions.


@click.command
@click.argument("vcfs", nargs=-1)
@click.option("-p", "--worker-processes", type=int, default=1)
@click.option("-f", "--force", is_flag=True, default=False)
# TODO add options for verbose and to force the use of a given
# index file
def cli(vcfs, worker_processes, force):
    data_path = pathlib.Path("validation-data")
    if len(vcfs) == 0:
        vcfs = (
            list(data_path.glob("*.vcf.gz"))
            + list(data_path.glob("*.bcf"))
            + list(data_path.glob("*.split"))
        )
    else:
        vcfs = [pathlib.Path(f) for f in vcfs]
    tmp_path = pathlib.Path("validation-tmp")
    tmp_path.mkdir(exist_ok=True)
    for f in vcfs:
        print("Validate", f)
        if f.is_dir():
            files = list(f.glob("*.vcf.gz")) + list(f.glob("*.bcf"))
            source_file = f.with_suffix("").with_suffix("")
        else:
            files = [f]
            source_file = f
        exploded = tmp_path / (f.name + ".exploded")
        if force and exploded.exists():
            shutil.rmtree(exploded)
        if not exploded.exists():
            vcf.explode(
                exploded,
                files,
                worker_processes=worker_processes,
                show_progress=True,
            )
        spec = tmp_path / (f.name + ".schema")
        if force or not spec.exists():
            with open(spec, "w") as specfile:
                vcf.mkschema(exploded, specfile)

        zarr = tmp_path / (f.name + ".zarr")
        if force and zarr.exists():
            shutil.rmtree(zarr)
        if not zarr.exists():
            vcf.encode(
                exploded,
                zarr,
                spec,
                worker_processes=worker_processes,
                show_progress=True,
            )

        vcf.validate(source_file, zarr, show_progress=True)


if __name__ == "__main__":
    cli()
