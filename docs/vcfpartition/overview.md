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
(sec-vcfpartition)=
# vcfpartition
```{code-cell} 
:tags: [remove-cell]
cp ../../tests/data/vcf/CEUTrio.20.21.gatk3.4.g.bcf* ./
cp ../../tests/data/vcf/NA12878.prod.chr20snippet.g.vcf.gz* ./
```

## Overview

The {ref}`cmd-vcfpartition` utility outputs a set of region strings
that partition indexed VCF/BCF files into either an approximate number of 
parts, or into parts of approximately a given size. This is useful 
for parallel processing of large VCF files.

:::{admonition} Why is this in bio2zarr?
The ``vcfpartition`` program is packaged with bio2zarr because the underlying
functionality was developed for {ref}`sec-vcf2zarr`, and there is currently 
no easy way to split processing of large VCFs up.
:::

### Partitioning into a number of parts

Here, we partition a BCF file into three parts using the ``--num-parts/-n`` 
argument:
```{code-cell}
vcfpartition CEUTrio.20.21.gatk3.4.g.bcf -n 3
```

The output is a tab-delimited stream of region strings and the file path.

:::{tip} 
The file path is included in the output to make it easy to work with 
multiple files at once, and also to simplify shell scripting tasks.
:::

We can use this, for example, in a shell loop to count the 
number of variants in each partition:

```{code-cell}
vcfpartition CEUTrio.20.21.gatk3.4.g.bcf -n 3 | while read split; 
do
    bcftools view -Hr $split | wc -l
done
```

:::{note}
Note that the number of variants in each partition is quite uneven, which
is generally true across files of all scales.
:::


Another important point is that there is granularity limit to the 
partitions:
```{code-cell}
vcfpartition CEUTrio.20.21.gatk3.4.g.bcf -n 30
```

Here, we asked for 30 partitions, but the underlying indexes provide
a maxmimum of 3.

:::{warning}
Do not assume that the number of partitions you ask for is what you get!
:::

### Partitioning into a fixed size

It is also possible to partition a VCF file into chunks of approximately
a given size. 


```{code-cell}
ls -lh  NA12878.prod.chr20snippet.g.vcf.gz
```

In this example, we have 3.8M file, and would like
to process this in chunks of approximately 500K at a time:

```{code-cell}
vcfpartition  NA12878.prod.chr20snippet.g.vcf.gz -s 500K
```

:::{tip}
Suffixes like M, MiB, G, GB, or raw numbers in bytes are all supported.
:::

We get 8 partitions in this example. Note again that these target sizes
are quite approximate.

### Parallel example

Here we use illustrate using `vcfpartition` to count the variants in each
partition in parallel using xargs. In this case we use 3 partitions with 3
processes, but because the number of variants per partition can be quite
uneven, it is a good idea to partition up work into (say) four times the number
of cores available for processing.

```{code-cell}
vcfpartition CEUTrio.20.21.gatk3.4.g.bcf -n 3 \
    | xargs -P 3 -I {} sh -c "bcftools view -Hr {} | wc -l"
```

