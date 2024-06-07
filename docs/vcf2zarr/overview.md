(sec-vcf2zarr)=
# vcf2zarr

Convert VCF data to the
[VCF Zarr specification](https://github.com/sgkit-dev/vcf-zarr-spec/)
reliably, in parallel or distributed over a cluster.

See the {ref}`sec-vcf2zarr-tutorial` for a step-by-step introduction
and the {ref}`sec-vcf2zarr-cli-ref` detailed documentation on
command line options.


## Quickstart

- First {ref}`install bio2zarr<sec-installation>`.


- Get some indexed VCF data:

```
curl -O https://raw.githubusercontent.com/sgkit-dev/bio2zarr/main/tests/data/vcf/sample.vcf.gz
curl -O https://raw.githubusercontent.com/sgkit-dev/bio2zarr/main/tests/data/vcf/sample.vcf.gz.tbi
```

- Convert to VCF Zarr in two steps:

```
vcf2zarr explode sample.vcf.gz sample.icf
vcf2zarr encode sample.icf sample.vcz
```

:::{tip}
If the ``vcf2zarr`` executable doesn't work, try ``python -m bio2zarr vcf2zarr``
instead.
:::

- Have a look at the results:

```
vcf2zarr inspect sample.vcz
```

### What next?

VCF Zarr is a starting point in what we hope will become a diverse ecosytem
of packages that efficiently process VCF data in Zarr format. However, this
ecosytem does not exist yet, and there isn't much software available
for working with the format. As such, VCF Zarr isn't suitable for end users
who just want to get their work done for the moment.

Having said that, you can:

- Look at the [VCF Zarr specification](https://github.com/sgkit-dev/vcf-zarr-spec/)
  to see how data is mapped from VCF to Zarr
- Use the mature [Zarr Python](https://zarr.readthedocs.io/en/stable/) package or
one of the other [Zarr implementations](https://zarr.dev/implementations/) to access
your data.
- Use the many functions in our [sgkit](https://sgkit-dev.github.io/sgkit/latest/)
sister project to analyse the data. Note that sgkit is under active development,
however, and the documentation may not be fully in-sync with this project.



## How does it work?
The conversion of VCF data to Zarr is a two-step process:

1. Convert ({ref}`explode<cmd-vcf2zarr-explode>`) VCF file(s) to
    Intermediate Columnar Format (ICF)
2. Convert ({ref}`encode<cmd-vcf2zarr-encode>`) ICF to Zarr

This two-step process allows `vcf2zarr` to determine the correct
dimension of Zarr arrays corresponding to each VCF field, and
to keep memory usage tightly bounded while writing the arrays.

:::{important}
The intermediate columnar format is not intended for any use
other than a temporary storage while converting VCF to Zarr.
The format may change between versions of `bio2zarr`.
:::

Both ``explode`` and ``encode`` can be performed in parallel
across cores on a single machine (via the ``--worker-processes`` argument)
or distributed across a cluster by the three-part ``init``, ``partition``
and ``finalise`` commands.

## Copying to object stores

:::{todo}
Document process of copying VCF Zarr datasets to an object store like S3.
See [Issue 234](https://github.com/sgkit-dev/bio2zarr/issues/234)
:::

