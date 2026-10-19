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

Alternatively supply an Oxford `.sample` sidecar (not a plain ID list):

```python
root = bgen.convert("input.bgen", "output.vcz", sample_file="input.sample")
# Explicitly use individual IDs instead of the default first identifier column.
root = bgen.convert(
    "input.bgen", sample_file="input.sample", sample_id_column="ID_2"
)
```

The first column is the default identifier, following QCTOOL semantics.
The one-column Oxford reference format and vcztools' `ID_1 ID_2 missing`
format are supported. Only IDs are imported, in unchanged BGEN sample order;
phenotype/covariate and `missing` columns are not imported or interpreted.
The file must contain exactly the BGEN sample count, with unique, nonmissing
selected IDs. Either explicit source overrides embedded identifiers.
`sample_ids` and `sample_file` are mutually exclusive; `sample_id_column`
requires `sample_file`. No sidecar is discovered automatically.
See {ref}`sec-bgen2zarr` for structural validation and missing-ID rules.

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

Dimension discovery is configurable independently of storage:

```python
# Default: actual sample-ploidy scan, discover phased/unphased records.
root = bgen.convert("input.bgen", dimension_source="actual")

# Allocate from declared ploidy bounds; loose bounds increase padding.
root = bgen.convert("input.bgen", dimension_source="declared")

# Independently known bound/model: no genotype-dimension discovery.
root = bgen.convert("input.bgen", max_ploidy=2, probability_model="unphased")

# Permit either phase model, deliberately allocating both GP and HP.
root = bgen.convert("input.bgen", max_ploidy=2, probability_model="mixed")
dimension_settings = root.attrs["bio2zarr_bgen_dimensions"]
```

`max_ploidy` accepts integers from 0 to 63 and is mutually exclusive with an
explicit `dimension_source`. `probability_model` defaults to `"auto"`; with
`max_ploidy` it still scans to discover which phase models are present.
Explicit `"unphased"` or `"phased"` assumptions and every provided ploidy bound
are checked during full decoding, including missing samples. Violations reject
conversion; no probability entries are truncated or silently discarded.
`"mixed"` always allocates both models. See {ref}`sec-bgen2zarr` for the
codec-specific discovery matrix and the backend release requirement.

## API reference

```{eval-rst}
.. autofunction:: bio2zarr.bgen.convert
```
