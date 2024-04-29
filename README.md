[![CI](https://github.com/sgkit-dev/bio2zarr/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sgkit-dev/bio2zarr/actions/workflows/ci.yml)

# bio2zarr
Convert bioinformatics file formats to Zarr

Initially supports converting VCF to the
[sgkit vcf-zarr specification](https://github.com/pystatgen/vcf-zarr-spec/)

**This is early alpha-status code: everything is subject to change,
and it has not been thoroughly tested**

## Install

```
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


## vcf2zarr


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

### Shell completion

To enable shell completion for a particular session in Bash do:

```
eval "$(_VCF2ZARR_COMPLETE=bash_source vcf2zarr)" 
```

If you add this to your ``.bashrc`` vcf2zarr shell completion should available
in all new shell sessions.

See the [Click documentation](https://click.palletsprojects.com/en/8.1.x/shell-completion/#enabling-completion)
for instructions on how to enable completion in other shells.
a

## plink2zarr

Convert a plink ``.bed`` file to zarr format. **This is incomplete**

## vcf_partition

Partition a given VCF file into (approximately) a give number of regions:

```
vcf_partition 20201028_CCDG_14151_B01_GRM_WGS_2020-08-05_chr20.recalibrated_variants.vcf.gz -n 10
```
gives
```
chr20:1-6799360
chr20:6799361-14319616
chr20:14319617-21790720
chr20:21790721-28770304
chr20:28770305-31096832
chr20:31096833-38043648
chr20:38043649-45580288
chr20:45580289-52117504
chr20:52117505-58834944
chr20:58834945-
```

These reqion strings can then be used to split computation of the VCF 
into chunks for parallelisation.

**TODO give a nice example here using xargs**

**WARNING that this does not take into account that indels may overlap 
partitions and you may count variants twice or more if they do**
