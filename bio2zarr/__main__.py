import click

from . import cli

@click.group()
def top_level():
    pass

# Provide a single top-level interface to all of the functionality.
# This probably isn't the recommended way of interacting, as we
# install individual commands as console scripts. However, this
# is handy for development and for those whose PATHs aren't set
# up in the right way.
top_level.add_command(cli.vcf2zarr)
top_level.add_command(cli.plink2zarr)

if __name__ == "__main__":
    top_level()
