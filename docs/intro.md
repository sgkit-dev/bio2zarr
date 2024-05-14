# bio2zarr

`bio2zarr` efficiently converts common bioinformatics formats to
[Zarr](https://zarr.readthedocs.io/en/stable/) format.

## Tools

- {ref}`sec-vcf2zarr` converts VCF data to
  [VCF Zarr](https://github.com/sgkit-dev/vcf-zarr-spec/) format.

- {ref}`sec-vcfpartition` is a utility to split an input (set of)
  VCFs into a given number of partitions. This is useful for
  parallel processing.

## Development status

`bio2zarr` is in development, contributions, feedback and issues are welcome
at the [GitHub repository](https://github.com/sgkit-dev/bio2zarr).

Support for converting PLINK data to VCF Zarr is partially implemented,
and adding BGEN support is also planned. If you would like to see
support for other formats (or an interested in helping with implementing),
please open an [issue on Github](https://github.com/sgkit-dev/bio2zarr/issues)
to discuss!

The package is currently focused on command line interfaces, but a
Python API is also planned.
