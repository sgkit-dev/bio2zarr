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

Select output fields and storage settings with the same defaults as the CLI:

```python
# Probabilities only, retaining the default float64 precision and zstd level 7.
root = bgen.convert("input.bgen", "probabilities.vcz", dosage=False)

# Smaller dosage-only output; this discards uncertainty and haplotype probabilities.
root = bgen.convert(
    "input.bgen",
    "dosage.vcz",
    probabilities=False,
    float_dtype="float32",
    compressor="lz4",
    compression_level=3,
    zarr_format=3,
)
settings = root.attrs["bio2zarr_bgen"]
```

`probabilities=True` and `dosage=True` retain both by default; disabling
both raises `ValueError` before input access or output creation.
`float_dtype` accepts only `"float64"` (default) or `"float32"`. Float32
rounds values and must not be treated as universally lossless; dosage is
computed in float64 before casting. Ploidy, missingness, the phase indicator
and variant/sample metadata remain in either mode. Codec settings apply to
schema-defined output arrays; auxiliary shared-writer arrays retain their
standard compression. See {ref}`sec-bgen2zarr` for full tradeoffs.

## API reference

```{eval-rst}
.. autofunction:: bio2zarr.bgen.convert
```
