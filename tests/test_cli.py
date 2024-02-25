import click.testing as ct

from bio2zarr import cli

# NOTE just putting things together here to see what works.
# Probably want to mock the module functions here to
# avoid testing any real functionality.
def test_vcf_summarise():
    runner = ct.CliRunner()
    result = runner.invoke(cli.vcf2zarr, "summarise", "filename")
    # FIXME not testing anything!
