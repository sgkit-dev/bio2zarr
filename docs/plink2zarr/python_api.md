(sec-plink2zarr-python-api)=
# Python API

Basic usage:
```python
import bio2zarr.plink as p2z

root = p2z.convert(plink_prefix, vcz_path)
```

This will convert the PLINK fileset with the given path prefix
(i.e. the shared prefix of the .bed, .bim, and .fam files)
to VCF Zarr stored at ``vcz_path``.

## API reference

```{eval-rst}

.. autofunction:: bio2zarr.plink.convert

```
