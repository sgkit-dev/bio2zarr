(sec-vcf2zarr-python-api)=
# Python API

Basic usage:
```python
import bio2zarr.vcf as v2z

v2z.convert([vcf_path], vcz_path)
```

To convert directly to an in-memory Zarr store (without writing to disk):
```python
root = v2z.convert([vcf_path])
```

To convert to a zip archive:
```python
root = v2z.convert([vcf_path], "output.vcz.zip")
```

For no-ICF conversion with a known genotype bound:
```python
root = v2z.convert([vcf_path], max_ploidy=2, batch_size=64)
root = v2z.convert([vcf_path], infer_shape=True, max_ploidy=2, max_alleles=4)
```

To infer array shapes and dtypes together in one metadata pass, and reuse
that discovery result:
```python
dimensions = v2z.scan_vcf_dimensions([vcf_path])
root = v2z.convert(
    [vcf_path], infer_shape=True, dimension_metadata=dimensions
)
```

For bounded discovery and reuse, repeat the output bound at conversion time:
```python
dimensions = v2z.scan_vcf_dimensions([vcf_path], max_alleles=4)
root = v2z.convert([vcf_path], max_alleles=4, dimension_metadata=dimensions)
```

Discovery validates `max_alleles` but returns observed field summaries; output
padding is chosen by `convert`. Reused metadata never disables conversion-time
allele-index, shape or dtype checks.

The metadata pass keeps summaries only; the conversion pass validates bounds
and preserves every field. Input files must remain unchanged between passes.
`max_alleles` counts REF (2 means biallelic) and may be supplied alone,
with `max_ploidy`, or with `infer_shape=True`. Supplied dimensions pad output;
actual overflows raise errors. Without `max_ploidy`, the metadata pass infers
ploidy too. Both bounds still require discovery of other fields and dtypes,
so they do not enable a single-pass conversion.
To choose temporary intermediate storage in the conventional path:
```python
root = v2z.convert(
    [vcf_path],
    temp_icf_dir="/path/to/temp",
)
```

See the overview for batch-memory semantics.

## API reference

```{eval-rst}

.. autofunction:: bio2zarr.vcf.convert
.. autofunction:: bio2zarr.vcf.scan_vcf_dimensions

```
