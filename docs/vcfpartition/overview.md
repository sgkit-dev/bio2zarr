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

## Overview

Partition a given VCF file into (approximately) a give number of regions:


```{code-cell}
vcfpartition CEUTrio.20.21.gatk3.4.g.bcf -n 3
```


```{code-cell}
vcfpartition CEUTrio.20.21.gatk3.4.g.bcf -n 3 \
    | xargs -P 3 -I {} sh -c "bcftools view -Hr {} CEUTrio.20.21.gatk3.4.g.bcf | wc -l"
```

