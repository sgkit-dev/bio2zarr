---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Bash
  language: bash
  name: bash
---
# vcf2zarr


## Overview


Convert a VCF to zarr format:

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



## Tutorial

This is a step-by-step tutorial showing you how to convert your
VCF data into Zarr format. There's three different ways to
convert your data, basically providing different levels of
convenience and flexibility corresponding to what you might
need for small, intermediate and large datasets.

### Small

<!-- ```{code-cell} bash -->
<!-- vcf2zarr convert ../tests/data/vcf/sample.vcf.gz sample.zarr -vf -->
<!-- ``` -->

 <div id="vcf2zarr_convert"></div>
 <script>
 AsciinemaPlayer.create('_static/vcf2zarr_convert.cast',
    document.getElementById('vcf2zarr_convert'), {
    cols:80,
    rows:12
 });
 </script>

### Intermediate

### Large
