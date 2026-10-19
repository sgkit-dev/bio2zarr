(sec-bgen2zarr)=
# bgen2zarr

`bgen2zarr` converts local BGEN genotype probability files to VCF Zarr using
the same schema, chunked writer, region indexing and parallel encoding pipeline
as the other bio2zarr converters. It does not depend on sgkit.

Install the optional backend with `python3 -m pip install 'bio2zarr[bgen]'`.
This installs [bgen-limix](https://github.com/jpfeuffer/bgen-limix), **not**
the legacy cbgen distribution. The maintained distribution intentionally
retains the Python import namespace `cbgen`. Dimension strategies require
the backend's public `read_dimensions` and compact `max_actual_ncomb` batch
interfaces; these are not present in the published 4.13.0 release.
Until a backend release includes these interfaces, this development version
requires a locally built backend wheel. See {ref}`sec-installation`
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
By default, GP or HP is omitted entirely if no records use that probability
model. Explicit dimension/model options can allocate larger padded widths or
both arrays, as described below.

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

A native metadata index provides bounded metadata reads. An existing
`input.bgen.metafile` sidecar is reused read-only; only when it is absent is
an index generated in a temporary directory. Existing sidecars are never
overwritten or silently regenerated. An older modification time or mismatched
variant count is rejected with an error. These checks are heuristics, not source
authentication: the native index stores no source fingerprint, and users must
ensure it belongs to the unchanged BGEN input. This is the bgen-limix native
metafile format, **not** the SQLite `.bgi` index produced by bgenix; a `.bgi`
does not replace a native metafile.

### Dimension discovery

The default, `--dimension-source actual --probability-model auto`, discovers
GP/HP dimensions from actual sample ploidies, including missing samples,
without decoding probabilities. Larger declared ploidy bounds do not inflate
native probability buffers or output padding.

Use `--dimension-source declared` to size from the record's declared maximum
instead. This can avoid scanning sample ploidies, at the cost of extra global
padding if annotations are loose. `--probability-model unphased|phased|mixed`
avoids discovering the phase model: unphased/phased is an assumption checked
during decoding, while mixed allocates both arrays even if only one is used.
Do not assume that all BGEN records share a phase model.

If a bound is known independently, `--max-ploidy 2` is an alternative to
`--dimension-source`; these options are mutually exclusive. With an explicit
model it removes genotype-dimension discovery altogether:

```bash
# Only for data independently known to be unphased with sample ploidy <= 2.
bgen2zarr convert input.bgen output.vcz --max-ploidy 2 --probability-model unphased

# Known ploidy bound, either or both phase models allowed; allocate GP and HP.
bgen2zarr convert input.bgen output.vcz --max-ploidy 2 --probability-model mixed

# Trust declared bounds for allocation, but discover which models are present.
bgen2zarr convert input.bgen output.vcz --dimension-source declared
```

These are allocation controls, not permission to truncate or discard data.
Actual ploidies (including missing samples) and explicit model assumptions
are checked during conversion. A violation fails conversion without publishing
the requested output. A human-genetics label alone is not evidence of a
diploid bound: sex chromosomes, unusual ploidy and mixed records still require
an appropriate bound/model.

The layout 2 discovery combinations are:

| Ploidy source | Probability model | Uncompressed | zlib | zstd |
| --- | --- | --- | --- | --- |
| actual | auto or explicit | Read sample-ploidy prefix | Decompress prefix and scan sample ploidies | Decompress full block and scan sample ploidies |
| declared | auto | Read declared bounds and seek phase | Decompress through phase, discard sample ploidies | Decompress full block; use declared bounds and phase |
| declared | explicit | Read first 8 uncompressed bytes | Decompress first 8 bytes | Decompress full block; use declared bounds |
| provided bound | auto | Discover phase | Decompress through phase | Decompress full block to discover phase |
| provided bound | explicit | No genotype-dimension read | No genotype-dimension read | No genotype-dimension read |

“Explicit” means unphased, phased or mixed. The provided-bound/auto mode
still discovers which models are present; it is not scan-free. Layout 1 has
fixed biallelic/diploid/unphased dimensions and needs no genotype payload for
dimension discovery, irrespective of compression. Its missingness is determined
when probabilities are decoded, not during discovery.

Prefix/header-only discovery is **not full payload validation**: it can miss
trailing corruption. Full decoding remains mandatory. zstd uses full block
decompression deliberately; streaming prefix reads were not consistently
faster in synthetic benchmarks. The table describes algorithms, not a
guaranteed speedup or a cohort-scale runtime prediction.

All output blocks share global GP/HP axes; they do not need the same actual
ploidy, allele count, or phase model. With a provided bound, conservative global
widths are calculated from that bound and indexed maximum allele count.
Native decoding checks actual widths against the selected capacity instead
of requiring buffers sized to a larger BGEN header declaration. Declared-source
mode still deliberately allocates its larger selected widths.
Index generation/variant metadata traversal, sample checks and
full probability decoding still occur in the scan-free mode.

The root attribute `bio2zarr_bgen_dimensions` records the resolved
`dimension_source` (`actual`, `declared`, or `provided`), `probability_model`,
and optional `max_ploidy`. Storage options remain in `bio2zarr_bgen`.

Conversion decodes a block of variants once into native float64 arrays.
Every selected field uses this same batch: GP/HP, DS, ploidy, missingness and
phase do not trigger separate genotype reads. Blocks stop at output variant-chunk
boundaries. Each output field is assembled and written one sample-chunk slab at
a time, using the existing writer's partition finalisation and region indexing.
There is no intermediate ICF probability store. The batch stream and output
writer are separate internally so a future ICF path need not change decoding;
no ICF mode or method-selection option is implemented yet.

An estimated **256 MiB per-worker working-buffer budget** limits native batch
and output slab allocations. Batches are subdivided within output chunks when
necessary, which can require repeated updates of a physical output chunk.
The budget includes conservative allowance for validation/temporary arrays,
but is not a hard process-RSS limit: native decompression, codecs, metadata and
Python overhead consume additional memory. Workers multiply resource use.
Native decoding still reads all samples of each variant. A variant that cannot
fit is rejected explicitly; reducing sample chunk size can reduce output slab
memory but cannot reduce this native all-samples lower bound. Extremely wide
polyploid/multiallelic records can make global GP/HP padding expensive.

Probability values are validated during the decode pass, not the metadata pass.
Directory/zip output is assembled in a temporary directory beside the destination
and renamed only after successful conversion. Invalid input therefore leaves no
partial output at the requested path. This stages the final output, not a second
decoded genotype representation. Temporary output and any generated temporary
index are removed on failure; pre-existing sidecars are left unchanged.
The input is not modified, and no generated native index or rewritten BGEN
file is placed alongside it.

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
