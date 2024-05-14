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
# Tutorial

This is a step-by-step tutorial showing you how to convert your
VCF data into Zarr format. There's three different ways to
convert your data, basically providing different levels of
convenience and flexibility corresponding to what you might
need for small, intermediate and large datasets.

## Small

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

## Intermediate

## Large

