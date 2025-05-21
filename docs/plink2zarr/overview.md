(sec-plink2zarr)=
# plink2zarr

Convert plink data to the
[VCF Zarr specification](https://github.com/sgkit-dev/vcf-zarr-spec/)
reliably in parallel.

See {ref}`sec-plink2zarr-cli-ref` for detailed documentation on
command line options.

Conversion of the plink data model to VCF follows the semantics of plink1.9 as closely
as possible. That is, given a binary plink fileset with prefix "fileset" (i.e.,
fileset.bed, fileset.bim, fileset.fam), running
```
$ plink2zarr convert fileset out.vcz
```
should produce the same result in ``out.vcz`` as
```
$ plink1.9 --bfile fileset --keep-allele-order --recode vcf-iid --out tmp
$ vcf2zarr convert tmp.vcf out.vcz
```

:::{warning}
It is important to note that we follow the same conventions as plink 2.0
where the A1 allele in the [bim file](https://www.cog-genomics.org/plink/2.0/formats#bim)
is the VCF ALT and A2 is the REF.
:::

:::{note}
Currently we only convert the basic VCF-like data from plink, and don't include
phenotypes and pedigree information. These are planned as future enhancements.
Please comment on [this issue](https://github.com/sgkit-dev/bio2zarr/issues/392)
if you are interested in this functionality.
:::




