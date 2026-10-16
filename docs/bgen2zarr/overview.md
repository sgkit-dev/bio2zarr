(sec-bgen2zarr)=
# bgen2zarr

`bgen2zarr` converts local BGEN genotype probability files to VCF Zarr using
the same schema, chunked writer, region indexing and parallel encoding pipeline
as the other bio2zarr converters. It does not depend on sgkit.

Install the optional backend with `python3 -m pip install 'bio2zarr[bgen]'`.
This installs [bgen-limix](https://github.com/jpfeuffer/bgen-limix), **not**
the legacy cbgen distribution. The maintained distribution intentionally
retains the Python import namespace `cbgen`. Version 4.12.0 or later is
required for native layout-1 indexing, safe decoding of every supported
encoding, and the public layout metadata API. See {ref}`sec-installation`
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

## Storage options

By default conversion retains both probabilities (GP and/or HP as applicable)
and dosage (DS), using **float64**, **Blosc zstd level 7 with byte shuffle**.
Choose the fields, precision and output compression independently:

```bash
# Probabilities only: preserve uncertainty and haplotype probabilities, omit DS.
bgen2zarr convert input.bgen probabilities.vcz --no-dosage

# Dosage only, explicitly rounding to float32 and using lz4.
bgen2zarr convert input.bgen dosage.vcz --no-probabilities --float-dtype float32 --compressor lz4 --compression-level 3
```

`--no-probabilities` omits both GP and HP; `--no-dosage` omits DS.
At least one must remain enabled. Omitted fields have no encoding pass, and
DS is not calculated when omitted. Original ploidy, missingness, per-record
phase indicator, and variant/sample metadata are always retained.
**Dosage-only output discards genotype uncertainty and haplotype
information**: ploidy and the phase indicator cannot reconstruct HP or GP.
No hard calls are invented in either mode.

`--float-dtype float32` rounds GP, HP and DS to 32-bit floats; this is not
universally lossless, especially for high-bit-depth BGEN probabilities.
Dosage is calculated from native float64 probabilities before rounding.
Neither precision renormalizes probabilities or reconstructs an omitted
state. Float64 preserves layout 1's independently rounded probabilities.
Both precisions use their exact dtype-specific VCZ missing/fill NaN bits.
Float16 is not supported by these BGEN conversion options.

`--compressor zstd|lz4` and `--compression-level 0..9` configure the
Blosc compressor on schema-defined output arrays, including GP/HP/DS.
They do not configure an intermediate ICF file or change the input BGEN
compression. Auxiliary arrays written by the shared writer (`sample_id`,
`contig_id`, and `region_index`) retain its standard compression.
The same options work with Zarr formats 2 and 3. Probability axes remain
whole within chunks; existing variant/sample chunk controls are unchanged.

The root attribute `bio2zarr_bgen` records `probabilities`, `dosage`,
`float_dtype`, `compressor`, and `compression_level`. The codec values name
the Blosc codec and level; byte shuffle is fixed. The selected configuration
is also logged with verbose logging (`-v`).

Embedded sample identifiers are used when present. Otherwise identifiers
`sample_0`, `sample_1`, ... are generated. Override either with a UTF-8
file containing exactly one unique nonempty identifier per line:

```bash
bgen2zarr convert input.bgen output.vcz --sample-ids sample_ids.txt
```

This is a simple ID list, **not** an Oxford `.sample` file: do not include a
header, type row, family identifiers or other columns.

## Native backend and resource use

The converter uses the public bgen-limix Python API for native indexing and
decoding of every supported layout and compression, including layout 1,
uncompressed layout 2 and records with no haplotypes. It does not rewrite
BGEN files into another encoding, parse genotype blocks in Python, or
depend on exported native C symbols or private native structures.

Layout 1's three independently rounded probabilities are decoded directly
as their original integers divided by 32768. Triplets whose sum is not
exactly one are retained without renormalization.

A temporary native metadata index provides bounded metadata reads. It is
removed on success or failure; no temporary copy of the genotype data is
created, and the input and its directory are not modified.

The converter scans all variant metadata and validates the data model before
creating output. It reads one complete variant's probabilities (all samples)
at a time, never the entire variant-by-sample matrix. Encoding uses the shared writer's
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

Only one local BGEN input is accepted per conversion. Remote access, Oxford
sample parsing, variant/sample selection, and hard-calling thresholds are
not provided. Zip output is read-only; use directory or memory output for
`mode="r+"`.

## Reference-data validation

The normal test suite includes small, vendored Oxford reference BGEN files
with matching independent GEN, VCF and HAPS data. It checks probabilities,
ordered haplotypes, metadata, ploidy, missingness and per-variant
dosage/frequency summaries in both Zarr formats. Checksums and the original
data-license notice are included in `tests/data/bgen`; CI requires no network
download. These are interoperability examples, not verified real-cohort data.
The example file's unsorted positions are tested explicitly, with a byte-
preserving sorted prefix used for successful conversion tests.
