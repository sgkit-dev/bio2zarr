(sec-vcf2zarr)=
# vcf2zarr

Convert VCF data to the
[VCF Zarr specification](https://github.com/sgkit-dev/vcf-zarr-spec/)
reliably, in parallel or distributed over a cluster.

See the {ref}`sec-vcf2zarr-tutorial` for a step-by-step introduction
and the {ref}`sec-vcf2zarr-cli-ref` detailed documentation on
command line options.


## Quickstart

First {ref}`install bio2zarr<sec-installation>`


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


## Common options

```
$ vcf2zarr convert <VCF1> <VCF2> <zarr>
```

Converts the VCF to zarr format.

**Do not use this for anything but the smallest files**

The recommended approach is to use a multi-stage conversion

First, convert the VCF into the intermediate format:

```
vcf2zarr explode tests/data/vcf/sample.vcf.gz tmp/sample.exploded
```

Then, (optionally) inspect this representation to get a feel for your dataset
```
vcf2zarr inspect tmp/sample.exploded
```

Then, (optionally) generate a conversion schema to describe the corresponding
Zarr arrays:

```
vcf2zarr mkschema tmp/sample.exploded > sample.schema.json
```

View and edit the schema, deleting any columns you don't want, or tweaking
dtypes and compression settings to your taste.

Finally, encode to Zarr:
```
vcf2zarr encode tmp/sample.exploded tmp/sample.zarr -s sample.schema.json
```

Use the ``-p, --worker-processes`` argument to control the number of workers used
in the ``explode`` and ``encode`` phases.

## To be merged with above

The simplest usage is:

```
$ vcf2zarr convert [VCF_FILE] [ZARR_PATH]
```


This will convert the indexed VCF (or BCF) into the vcfzarr format in a single
step. As this writes the intermediate columnar format to a temporary directory,
we only recommend this approach for small files (< 1GB, say).

The recommended approach is to run the conversion in two passes, and
to keep the intermediate columnar format ("exploded") around to facilitate
experimentation with chunk sizes and compression settings:

```
$ vcf2zarr explode [VCF_FILE_1] ... [VCF_FILE_N] [ICF_PATH]
$ vcf2zarr encode [ICF_PATH] [ZARR_PATH]
```

The inspect command provides a way to view contents of an exploded ICF
or Zarr:

```
$ vcf2zarr inspect [PATH]
```

This is useful when tweaking chunk sizes and compression settings to suit
your dataset, using the mkschema command and --schema option to encode:

```
$ vcf2zarr mkschema [ICF_PATH] > schema.json
$ vcf2zarr encode [ICF_PATH] [ZARR_PATH] --schema schema.json
```

By editing the schema.json file you can drop columns that are not of interest
and edit column specific compression settings. The --max-variant-chunks option
to encode allows you to try out these options on small subsets, hopefully
arriving at settings with the desired balance of compression and query
performance.

