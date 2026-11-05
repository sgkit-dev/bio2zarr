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
```

To infer ploidy together with other dimensions in one metadata pass, and reuse
that discovery result:
```python
dimensions = v2z.scan_vcf_dimensions([vcf_path])
root = v2z.convert(
    [vcf_path], infer_ploidy=True, dimension_metadata=dimensions
)
```

The metadata pass keeps summaries only; the conversion pass validates bounds
and preserves every field. Input files must remain unchanged between passes.
To choose temporary intermediate storage in the conventional path:
```python
root = v2z.convert(
    ["s3://my-bucket/input/sample.vcf.gz"],
    temp_icf_dir="s3://my-bucket/tmp",
)
```

See the overview for S3 credentials, endpoint configuration and batch-memory
semantics.

## API reference

```{eval-rst}

.. autofunction:: bio2zarr.vcf.convert
.. autofunction:: bio2zarr.vcf.scan_vcf_dimensions

```
