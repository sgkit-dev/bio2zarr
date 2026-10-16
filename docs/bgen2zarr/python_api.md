(sec-bgen2zarr-python-api)=
# Python API

```python
from bio2zarr import bgen

root = bgen.convert("input.bgen", "output.vcz")
probabilities = root["call_GP"]
dosages = root["call_DS"]
```

`call_GP` is present for unphased records and `call_HP` for phased records;
mixed input produces both. Phasing is not converted to hard genotype calls.
`call_PLOIDY`, `call_BGEN_MISSING` and `variant_bgen_phased` retain the
original data model and missingness. Multiallelic input adds an
`alt_alleles` dimension to `call_DS`; entirely biallelic input retains
two-dimensional dosage. See {ref}`sec-bgen2zarr` for probability ordering,
padding and exact layout-1 preservation.

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
