# bio2zarr

`bio2zarr` efficiently converts common bioinformatics formats to
[Zarr](https://zarr.readthedocs.io/en/stable/) format.

To convert
the resulting  [VCF Zarr](https://github.com/sgkit-dev/vcf-zarr-spec/)
files to other formats, see [vcztools](https://github.com/sgkit-dev/vcztools).

## Tools

- {ref}`sec-vcf2zarr` converts VCF data to
  [VCF Zarr](https://github.com/sgkit-dev/vcf-zarr-spec/) format.

- {ref}`sec-plink2zarr` converts PLINK 1.0 data to
  [VCF Zarr](https://github.com/sgkit-dev/vcf-zarr-spec/) format.

- {ref}`sec-tskit2zarr` converts [tskit](https://tskit.dev)
  data into [VCF Zarr](https://github.com/sgkit-dev/vcf-zarr-spec/) format.

- {ref}`sec-vcfpartition` is a utility to split an input
  VCF into a given number of partitions. This is useful for
  parallel processing of VCF data.

- {ref}`sec-zipzarr` is a utility for zipping/unzipping Zarr directory
  stores. There can be significant advantages to using a single
  zipped VCZ over a large directory tree, and this utility makes it
  straightforward to switch between the two representations.

## Development status

`bio2zarr` is in development, contributions, feedback and issues are welcome
at the [GitHub repository](https://github.com/sgkit-dev/bio2zarr).

If you would like to see
support for other formats such as BGEN (or an interested in helping with implementing),
please open an [issue on Github](https://github.com/sgkit-dev/bio2zarr/issues)
to discuss!

## Python APIs

There is access to some limited functionality via Python APIs (documented
along with the respective tools). These are in beta, and should be fully
documented and stabilised in the coming releases. General APIs to enable
efficient and straightforward encoding of data to VCZ are planned
(see [issue #412](https://github.com/sgkit-dev/bio2zarr/issues/412)).

