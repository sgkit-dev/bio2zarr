import click

from . import cli

@cli.version
@click.group()
def bio2zarr():
    pass

# Provide a single top-level interface to all of the functionality.
# This probably isn't the recommended way of interacting, as we
# install individual commands as console scripts. However, this
# is handy for development and for those whose PATHs aren't set
# up in the right way.
bio2zarr.add_command(cli.vcf2zarr)
bio2zarr.add_command(cli.plink2zarr)

if __name__ == "__main__":
    bio2zarr()
