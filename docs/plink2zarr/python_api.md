(sec-plink2zarr-python-api)=
# Python API

Basic usage:
```python
import bio2zarr.plink as p2z

root = p2z.convert(plink_prefix, vcz_path)
```

This will convert the PLINK fileset with the given path prefix
(i.e. the shared prefix of the .bed/.bim/.fam or .pgen/.pvar/.psam files)
to VCF Zarr stored at ``vcz_path``.

For PLINK 2 with unsplit male X calls:

```python
root = p2z.convert(
    plink_prefix,
    vcz_path,
    genome_build="hg38",
    worker_processes=4,
    variants_chunk_size=1000,
    samples_chunk_size=1000,
)
```

Install `bio2zarr[plink2]`. See {ref}`sec-plink2zarr` for ploidy, phasing,
and dosage conversion semantics. DS/HDS are included by default; use
`dosages=False` for explicitly hardcall-only conversion.

## API reference

```{eval-rst}

.. autofunction:: bio2zarr.plink.convert

```
