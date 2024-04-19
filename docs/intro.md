# bio2zarr Documentation

`bio2zarr` efficiently converts common bioinformatics formats to 
[Zarr](https://zarr.readthedocs.io/en/stable/) format. Initially supporting converting 
VCF to the [sgkit vcf-zarr specification](https://github.com/pystatgen/vcf-zarr-spec/).

`bio2zarr` is in early alpha development, contributions, feedback and issues are welcome
at the [GitHub repository](https://github.com/sgkit-dev/bio2zarr).

## Installation
`bio2zarr` can be installed from PyPI using pip:

```bash
$ python3 -m pip install bio2zarr
```

This will install the programs ``vcf2zarr``, ``plink2zarr`` and ``vcf_partition``
into your local Python path. You may need to update your $PATH to call the 
executables directly.

Alternatively, calling 
```
$ python3 -m bio2zarr vcf2zarr <args>
```
is equivalent to 

```
$ vcf2zarr <args>
```
and will always work.

## Basic vcf2zarr usage
For modest VCF files (up to a few GB), a single command can be used to convert a VCF file
(or set of VCF files) to Zarr:

```bash
$ vcf2zarr convert <VCF1> <VCF2> ... <VCFN> <zarr>
```

For larger files a multi-step process is recommended. 


First, convert the VCF into the intermediate format:

```bash
$ vcf2zarr explode tests/data/vcf/sample.vcf.gz tmp/sample.exploded
```

Then, (optionally) inspect this representation to get a feel for your dataset
```bash
$ vcf2zarr inspect tmp/sample.exploded
```

Then, (optionally) generate a conversion schema to describe the corresponding
Zarr arrays:

```bash
$ vcf2zarr mkschema tmp/sample.exploded > sample.schema.json
```

View and edit the schema, deleting any columns you don't want, or tweaking 
dtypes and compression settings to your taste.

Finally, encode to Zarr:
```bash
$ vcf2zarr encode tmp/sample.exploded tmp/sample.zarr -s sample.schema.json
```

Use the ``-p, --worker-processes`` argument to control the number of workers used
in the ``explode`` and ``encode`` phases.




```{tableofcontents}
```
