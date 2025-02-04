(sec-vcf2zarr)=
# vcf2zarr

Convert VCF data to the
[VCF Zarr specification](https://github.com/sgkit-dev/vcf-zarr-spec/)
reliably, in parallel or distributed over a cluster.

See the {ref}`sec-vcf2zarr-tutorial` for a step-by-step introduction
and the {ref}`sec-vcf2zarr-cli-ref` detailed documentation on
command line options.

See the [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2024.06.11.598241) for 
further details.

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
of packages that efficiently process VCF data in Zarr format. This 
ecosytem is in its infancy and there isn't much software available
for performing off-the-shelf bioinformatics tasks
working with the format. As such, VCF Zarr isn't suitable for end users
who just want to get their work done for the moment, and is currently
aimed methods developers and early adopters.

Having said that, you can:

- Use [vcztools](https://github.com/sgkit-dev/vcztools/) as a drop-in replacment 
  for bcftools, transparently using Zarr on local storage or cloud stores as the 
  backend.
- Look at the [VCF Zarr specification](https://github.com/sgkit-dev/vcf-zarr-spec/)
  to see how data is mapped from VCF to Zarr
- Use the mature [Zarr Python](https://zarr.readthedocs.io/en/stable/) package or
one of the other [Zarr implementations](https://zarr.dev/implementations/) to access
your data.
- Use the many functions in our [sgkit](https://sgkit-dev.github.io/sgkit/latest/)
sister project to analyse the data. Note that sgkit is under active development,
however, and the documentation may not be fully in-sync with this project.

For more information, please see our 
bioRxiv preprint [Analysis-ready VCF at Biobank scale using Zarr](
https://www.biorxiv.org/content/10.1101/2024.06.11.598241).


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

## Local alleles

As discussed in our [preprint](
https://www.biorxiv.org/content/10.1101/2024.06.11.598241) 
vcf2zarr has an experimental implementation of the local alleles data
reduction technique. This essentially reduces the inner dimension of 
large fields such as AD by storing information relevant only to the alleles
involved in a particular variant call, rather than information information
for all alleles. This can make a substantial difference when there is a large 
number of alleles.

To use local alleles, you must generate storage a schema (see the 
{ref}`sec-vcf2zarr-tutorial-medium-dataset` section of the tutorial)
using the {ref}`mkschema<cmd-vcf2zarr-mkschema>` command with the 
``--local-alleles`` option. This will generate the ``call_LA`` field
which lists the alleles observed for each genotype call, and 
translate supported fields from their global alleles to local
alleles representation.

:::{warning}
Support for local-alleles is preliminary and may be subject to change
as the details of how alleles for a particular call are chosen, and the 
number of alleles retained determined. Please open an issue on
[GitHub](https://github.com/sgkit-dev/bio2zarr/issues/) if you would like to 
help improve Bio2zarr's local alleles implementation.
:::

:::{note}
Only the PL and AD fields are currently supported for local alleles
data reduction. Please comment on our 
[local alleles fields tracking issue](https://github.com/sgkit-dev/bio2zarr/issues/315)
if you would like to see other fields supported, or to help out with 
implementing more.
:::

## Debugging

When things go wrong with conversion, it's very useful to generate some 
debugging information using the ``-v`` or ``-vv`` options. These can 
help identify what's going wrong (usually running out of memory).

:::{warning}
To get the full logging output you **must use** -p0, such that no multiprocessing
is used. This means that tracking down problems can be slow, unfortunately.
This is due to a bug in the way logging from child processes is handled;
see [issue 302](https://github.com/sgkit-dev/bio2zarr/issues/302) for details.
:::



## Copying to object stores

:::{todo}
Document process of copying VCF Zarr datasets to an object store like S3.
See [Issue 234](https://github.com/sgkit-dev/bio2zarr/issues/234)
:::

