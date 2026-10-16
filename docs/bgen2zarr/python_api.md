(sec-bgen2zarr-python-api)=
# Python API

```python
from bio2zarr import bgen

root = bgen.convert("input.bgen", "output.vcz")
probabilities = root["call_GP"]
dosages = root["call_DS"]
```

Omit the output path for a MemoryStore-backed group; use a `.zip` output
suffix for a ZipStore-backed group. All output modes return a `zarr.Group`.
Custom sample identifiers may be supplied directly:

```python
root = bgen.convert(
    "input.bgen",
    sample_ids=["sample_a", "sample_b"],
    variants_chunk_size=100,
    samples_chunk_size=1000,
    zarr_format=3,
)
```

The number of identifiers must match the BGEN sample count.
See {ref}`sec-bgen2zarr` for the supported data model and limitations.

## API reference

```{eval-rst}
.. autofunction:: bio2zarr.bgen.convert
```
