(sec-bgen2zarr-cli-ref)=
# CLI Reference

Defaults keep probabilities and dosage in float64 using Blosc zstd level 7
with byte shuffle. For probabilities only, use `--no-dosage`. For dosage-only
float32 output, use `--no-probabilities --float-dtype float32`; this discards
uncertainty and haplotype probabilities and explicitly rounds dosage.
`--compressor lz4 --compression-level 3` changes output compression, not the
input BGEN compression. See {ref}`sec-bgen2zarr` for examples and tradeoffs.

```{eval-rst}
.. _cmd-bgen2zarr-convert:
.. click:: bio2zarr.cli:convert_bgen
   :prog: bgen2zarr convert
   :nested: full
```
