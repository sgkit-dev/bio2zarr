# Development script to automate generating multi-contig
# VCFs of various lenghts for testing the indexing code.
# TODO should move to a "scripts" directory or something
import sys

import click


def write_header(num_contigs):
    click.echo("##fileformat=VCFv4.2")
    click.echo(f"##source={' '.join(sys.argv)}")
    click.echo('##FILTER=<ID=PASS,Description="All filters passed">')
    for contig in range(num_contigs):
        click.echo(f"##contig=<ID={contig}>")
    header = "\t".join(["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"])
    click.echo(header)


@click.command
@click.argument("contigs", type=int)
@click.argument("rows-per-contig", type=int)
def cli(contigs, rows_per_contig):
    write_header(contigs)
    for j in range(contigs):
        for k in range(rows_per_contig):
            pos = str(k + 1)
            click.echo("\t".join([str(j), pos, "A", "T", ".", ".", ".", "."]))


if __name__ == "__main__":
    cli()
