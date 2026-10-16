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

Supported input is **layout 1** (uncompressed or zlib) and **layout 2**
(uncompressed, zlib or zstd). Layout 2 supports **phased and unphased
probabilities, multiallelic variants, and variable sample ploidy from 0 to 63**,
with 1 to 32 bits per probability. These models can vary between records in
the same file. Layout 1 is biallelic, diploid and unphased by definition.
Allele order is retained: the first BGEN allele is interpreted as reference
and subsequent alleles as alternates. BGEN does not guarantee which allele is the
reference genome allele; callers must establish the intended orientation.
Positions remain one-based. Contigs retain their names in first-seen order.
No genome assembly or contig lengths are invented.

| Output | Meaning |
| --- | --- |
| `call_GP` | Unphased genotype probabilities in VCF/BGEN colexicographic order |
| `call_HP` | Phased haplotype probabilities, flattened haplotype-major, then allele order |
| `call_DS` | Expected count of each alternate allele, using the sample's ploidy |
| `call_PLOIDY` | Original per-variant, per-sample ploidy, including missing samples |
| `call_BGEN_MISSING` | Original per-variant, per-sample missingness flag (0 or 1) |
| `variant_bgen_phased` | Whether each record contains phased rather than unphased probabilities |
| `variant_id`, `variant_id_mask` | BGEN rsID; empty or `.` rsIDs are missing |
| `variant_bgen_id` | Separate BGEN variant identifier, retained without replacing rsID |
| `variant_position`, `variant_contig`, `variant_allele`, `variant_length` | Position, contig index, alleles and reference-allele length |
| `sample_id` | Sample identifiers in the original file order |

GP, HP and DS are float64 to retain the reader's precision, including 32-bit
BGEN probabilities and layout 1's independently rounded probabilities.
Probabilities are not renormalized or requantized.

GP's dimensions are `(variants, samples, genotypes)`; its width is the maximum
observed `choose(ploidy + alleles - 1, ploidy)`. For diploid biallelic records,
the order is 0/0, 0/1, 1/1. For diploid triallelic records it is 0/0, 0/1,
1/1, 0/2, 1/2, 2/2. HP's dimensions are
`(variants, samples, FORMAT_HP_dim)`; its width is the maximum observed
`ploidy * alleles` (at least 1). For diploid biallelic records HP contains
`[P(haplotype 0, allele 0), P(haplotype 0, allele 1),
P(haplotype 1, allele 0), P(haplotype 1, allele 1)]`.
GP or HP is omitted entirely if no records use that probability model.

For an entirely biallelic input DS has dimensions `(variants, samples)`.
If any record is multiallelic, DS has dimensions
`(variants, samples, alt_alleles)`, in original alternate-allele order.
Unphased DS weights each genotype probability by its allele counts;
phased DS sums each allele's probability across haplotypes.
For diploid biallelic unphased data this is GP[1] + 2 * GP[2].

Missing samples have VCZ's float64 missing-NaN sentinel in their active
probability entries and DS, not zero probabilities or zero dosage.
Unused probability slots and nonexistent alternate alleles use VCZ's
distinct fill-NaN sentinel. For a record of the other probability model,
GP or HP starts with a missing sentinel followed by fill sentinels.
Zero-ploidy samples have no HP entries (all padding) and, when present,
dosage zero; `call_PLOIDY` and `call_BGEN_MISSING` preserve the distinction
between an empty vector and a missing call.
No hard genotype inference or fabricated `call_genotype`/GT phasing arrays
are introduced: phasing is retained in HP and `variant_bgen_phased`.

Embedded sample identifiers are used when present. Otherwise identifiers
`sample_0`, `sample_1`, ... are generated. Override either with a UTF-8
file containing exactly one unique nonempty identifier per line:

```bash
bgen2zarr convert input.bgen output.vcz --sample-ids sample_ids.txt
```

This is a simple ID list, **not** an Oxford `.sample` file: do not include a
header, type row, family identifiers or other columns.

## Backend compatibility and resource use

The converter uses the public bgen-limix Python API, including on Windows;
it does not depend on exported native C symbols or private native structures.
Small compatibility adapters avoid defects in the 4.11 backend:

- Uncompressed layout 2 is staged as a temporary zlib-compressed BGEN file.
  The original probability bits and metadata are unchanged.
- Layout 1 uses a temporary, on-disk metadata index and layout-2 transport
  file. Each original uint16 probability integer is carried unchanged in a
  four-state haploid vector decoded by bgen-limix. The three original
  integers are then recovered exactly and divided by 32768; the fourth
  transport state is discarded. Original alleles and diploid/unphased
  metadata remain authoritative. This avoids lossy conversion to a
  different probability denominator, including for independently rounded
  triplets whose sum is not exactly one.
- Phased records with all ploidies zero contain no encoded probabilities.
  Their ploidy/missingness metadata is read directly, avoiding the native
  decoder's rejection of zero maximum ploidy.

Staging reads one variant at a time and requires temporary disk space for
the compressed copy as well as the metadata index. Temporary files are
removed on success or failure; the input and its directory are not modified.

The converter scans all variant metadata and validates the data model before
creating output. It reads one complete variant's probabilities (all samples)
at a time, never the entire variant-by-sample matrix. The temporary index
provides bounded metadata reads. Encoding uses the shared writer's
variant-chunk buffers across all samples; sample chunk size controls storage,
not native decode size. Worker processes multiply the per-worker buffer usage.
Fields are encoded in separate streaming passes. Multiallelic/polyploid
unphased data can have very large genotype dimensions; output padding and
buffer sizes depend on the maximum observed dimension, not only chunk sizes.

## Explicit limitations

Empty files, invalid layouts/compression, missing alleles/contigs,
nonpositive or decreasing positions within a chromosome, and invalid
probabilities are rejected. Input is not silently reordered. Layout 1
does not define zstd compression; layout 2 does not define compression flag 3.
The backend limits sample and variant counts to `2**31 - 1`.

Only one local BGEN input is accepted per conversion. Remote access, Oxford
sample parsing, variant/sample selection, and hard-calling thresholds are
not provided. Zip output is read-only; use directory or memory output for
`mode="r+"`.
