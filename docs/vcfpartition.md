(xxx_vcfpartition)=
# vcfpartition xxx

## Overview

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

**WARNING that this does not take into account that indels may overlap**


## CLI Reference

FIXME not working due to

```{eval-rst}
.. click:: bio2zarr.cli:vcfpartition
   :prog: vcfpartition
   :nested: full
```
