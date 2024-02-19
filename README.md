# bio2zarr
Convert bioinformatics file formats to Zarr

Initially supports converting VCF to the
[sgkit vcf-zarr specification](https://github.com/pystatgen/vcf-zarr-spec/)

**This is early alpha-status code: everything is subject to change, a
and it has not been thoroughly tested**

## Usage

Convert a VCF to zarr format:

```
python3 -m bio2zarr vcf2zarr convert <VCF> <zarr>
```

Converts the VCF to zarr format.

**Do not use this for anything but the smallest files**

The recommended approach is to use a multi-stage conversion

First, convert the VCF into an intermediate columnar format:

```
python3 -m bio2zarr vcf2zarr convert tests/data/vcf/sample.vcf.gz tmp/sample.exploded
```

Then, (optionally) inspect this representation to get a feel for your dataset
```
python3 -m bio2zarr vcf2zarr summarise tmp/sample.exploded
```

Then, (optionally) generate a conversion schema to describe the corresponding
Zarr arrays:

```
python3 -m bio2zarr vcf2zarr genspec tmp/sample.exploded > sample.schema.json
```

View and edit the schema, deleting any columns you don't want.

Finally, convert to Zarr

```
python3 -m bio2zarr vcf2zarr to-zarr tmp/sample.exploded tmp/sample.zarr -s sample.schema.json
```

Use the ``-p, --worker-processes`` argument to control the number of workers used
to do zarr encoding.


