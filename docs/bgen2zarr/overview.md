(sec-bgen2zarr)=
# bgen2zarr

`bgen2zarr` converts local BGEN genotype probability files to VCF Zarr using
the same schema, chunked writer, region indexing and parallel encoding pipeline
as the other bio2zarr converters. It does not depend on sgkit.

Install the optional backend with `python3 -m pip install 'bio2zarr[bgen]'`.
This installs [bgen-limix](https://github.com/jpfeuffer/bgen-limix), **not**
the legacy cbgen distribution. The maintained distribution intentionally
retains the Python import namespace `cbgen`. See {ref}`sec-installation`
for platform requirements.

```bash
bgen2zarr convert input.bgen output.vcz
bgen2zarr convert input.bgen output.vcz.zip --zarr-format 3
bgen2zarr convert input.bgen output.vcz --variants-chunk-size 100 --samples-chunk-size 1000
```

`python3 -m bio2zarr bgen2zarr` is equivalent to `bgen2zarr`.
The output defaults to Zarr format 2, or the value of
`BIO2ZARR_DEFAULT_ZARR_FORMAT`; use `--zarr-format 2` or `--zarr-format 3`
to override it. Existing output paths require confirmation (or `--force`).

## Data model

Supported input is **unphased, diploid, biallelic layout 2** (BGEN 1.2/1.3),
compressed with **zlib or zstd**, with 1 to 32 bits per probability.
Allele order is retained: the first BGEN allele is interpreted as reference
and the second as alternate. BGEN does not guarantee which allele is the
reference genome allele; callers must establish the intended orientation.
Positions remain one-based. Contigs retain their names in first-seen order.
No genome assembly or contig lengths are invented.

| Output | Meaning |
| --- | --- |
| `call_GP` | Genotype probabilities in VCF order: 0/0, 0/1, 1/1 |
| `call_DS` | Expected alternate-allele count: GP[1] + 2 * GP[2] |
| `variant_id`, `variant_id_mask` | BGEN rsID; empty or `.` rsIDs are missing |
| `variant_bgen_id` | Separate BGEN variant identifier, retained without replacing rsID |
| `variant_position`, `variant_contig`, `variant_allele`, `variant_length` | Position, contig index, alleles and reference-allele length |
| `sample_id` | Sample identifiers in the original file order |

GP and DS are float64 to retain the reader's precision, including 32-bit BGEN
probabilities. Missing samples have VCZ's float64 missing-NaN sentinel in
all GP entries and in DS, not zero probabilities or zero dosage.
No quantization, probability packing, hard genotype inference or
`call_genotype`/phasing arrays are introduced.

Embedded sample identifiers are used when present. Otherwise identifiers
`sample_0`, `sample_1`, ... are generated. Override either with a UTF-8
file containing exactly one unique nonempty identifier per line:

```bash
bgen2zarr convert input.bgen output.vcz --sample-ids sample_ids.txt
```

This is a simple ID list, **not** an Oxford `.sample` file: do not include a
header, type row, family identifiers or other columns.

## Explicit limitations

Phased haplotype probabilities, non-diploid samples (including mixed ploidy),
and multiallelic variants are supported by the backend but are **outside this
converter's initial scope**. Phased probabilities must not be mislabeled as
unphased GP; supporting them requires a haplotype-probability schema.
Non-diploid/multiallelic conversion requires ploidy-aware genotype dimensions
and dosage conventions beyond the three-genotype scalar-dosage model above.
These cases raise errors rather than being dropped or coerced.

Layout 1 is rejected because bgen-limix 4.11's variant indexing requires
layout 2. Uncompressed layout 2 is rejected because that backend's
uncompressed header reader uses a fixed-size buffer rather than the
variable-length layout 2 block. Both are rejected **before native file
access**, even though the backend contains some code for these encodings.
Use compressed layout 2 input instead. Empty files, missing alleles/contigs,
nonpositive or decreasing positions within a chromosome, and invalid
probabilities are also rejected. Input is not silently reordered.

The converter scans all variant metadata and validates the data model before
creating output. It reads one complete variant's probabilities (all samples)
at a time, never the entire variant-by-sample matrix. A temporary metafile
index provides bounded partition reads and is removed on success or failure;
the input directory is not modified. Encoding uses the shared writer's
variant-chunk buffers across all samples; sample chunk size controls storage,
not native decode size. Worker processes multiply the per-worker buffer usage.
GP and DS are encoded in separate streaming passes.

Only one local BGEN input is accepted per conversion. Remote access, Oxford
sample parsing, variant/sample selection, and hard-calling thresholds are
not provided. Zip output is read-only; use directory or memory output for
`mode="r+"`.
