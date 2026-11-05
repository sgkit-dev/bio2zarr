(sec-plink2zarr)=
# plink2zarr

Convert plink data to the
[VCF Zarr specification](https://github.com/sgkit-dev/vcf-zarr-spec/)
reliably in parallel.

Both PLINK 1 (`.bed/.bim/.fam`) and PLINK 2 (`.pgen/.pvar/.psam`)
filesets are supported. PLINK 2 requires the optional
[pgenlib](https://github.com/chrchang/plink-ng/tree/master/2.0/Python) bindings:

```bash
python -m pip install 'bio2zarr[plink2]'
plink2zarr convert cohort out.vcz --worker-threads 4 \
    --variants-chunk-size 1000 --samples-chunk-size 1000
```

PGEN conversion is supported on Linux and macOS; upstream pgenlib does not
currently provide Windows wheels.

Supply the shared prefix, not a filename. `.pvar.zst` is also supported.
The genotype file selects the format automatically; use `--input-format bed`
or `--input-format pgen` when both formats share a prefix.

See {ref}`sec-plink2zarr-cli-ref` for detailed documentation on
command line options.

## PLINK 1

Conversion of the PLINK 1 data model to VCF follows the semantics of plink1.9 as closely
as possible. That is, given a binary plink fileset with prefix "fileset" (i.e.,
fileset.bed, fileset.bim, fileset.fam), running
```
$ plink2zarr convert fileset out.vcz
```
should produce the same result in ``out.vcz`` as
```
$ plink1.9 --bfile fileset --keep-allele-order --recode vcf-iid --out tmp
$ vcf2zarr convert tmp.vcf out.vcz
```

:::{warning}
It is important to note that we follow the same conventions as plink 2.0
where the A1 allele in the [bim file](https://www.cog-genomics.org/plink/2.0/formats#bim)
is the VCF ALT and A2 is the REF.
:::

## PLINK 2

REF and comma-separated ALT alleles are taken directly from PVAR, including
multiallelic sites. Sample IDs are PSAM IIDs; they must be unique even when FIDs
differ. Variant order is retained and must be sorted by position within
contiguous chromosome groups.

### Ploidy and phase

PGEN encodes haploid hardcalls as duplicated alleles: it cannot distinguish a
haploid `A` from diploid `A/A`. By default, the converter reconstructs **human**
ploidy from chromosome labels and PSAM `SEX` (`1` male, `2` female).
Autosomes and PAR are diploid; male non-PAR X, male Y, and mitochondrial
chromosomes are haploid. Female Y has zero copies and must contain missing calls.
Standard labels (`X`, `Y`, `MT`, `M`, `XY`, `PAR1`, `PAR2`), their `chr`-prefixed
forms, and human numeric chromosome codes 23-26 are recognized.

For unsplit X with male samples, specify the reference genome:

```bash
plink2zarr convert cohort out.vcz --genome-build hg38 --worker-threads 4
```

`hg19` and `hg38` use their respective one-based, inclusive PAR intervals.
If the fileset contains explicit `PAR1`/`PAR2` chromosome labels from PLINK's
`--split-par`, remaining X variants are treated as non-PAR without requiring a
build. The converter does not infer a build from variant coordinates.
Missing/unknown sex on X/Y, missing PAR information for male X, heterozygous
haploid calls, and nonmissing female Y calls are errors rather than guesses.

For other organisms, custom chromosome sets, or to preserve pgenlib's
two-allele representation without ploidy reconstruction, use
`--ploidy-mode diploid`. Original ploidy cannot be recovered in that mode.
The `##chrSet` PVAR header is not interpreted.

Output ploidy and allele dimensions are fixed from metadata before genotype
decoding. Shorter calls are padded with VCZ's `-2` fill value; missing alleles
are `-1`. `call_genotype_mask` marks both negative values. The ploidy axis has
length one for entirely haploid datasets and two when any diploid call is
possible from metadata.

Heterozygous hardcall phasing and allele order are preserved. PGEN does not
retain original homozygous/missing-call phase flags; when phase information is
present, these flags follow pgenlib's representation. Without file-wide
hardcall phase information, all phase flags are false. Haploid/zero-copy phase
flags are always false.

### Dosages

By default, conversion also exports `call_DS` (expected alternate allele
counts) and `call_HDS` (alternate allele counts per haplotype), using
`pgenlib>=0.95.1`. Stored fractional dosages are preserved, including when the
hardcall is missing. Hardcalls are never inferred from dosage values.

When no explicit dosages are stored, pgenlib supplies values derived from the
hardcalls. Unphased dosages are split evenly between the two haplotypes;
hardcall-phase information is used when available, and explicitly stored phased
dosages retain their haplotype values. HDS values do not imply that hardcalls
were phased: `call_genotype_phased` describes hardcalls only.

PGEN uses the diploid dosage scale even for haploid chromosomes. Metadata mode
normalizes haploid DS from 0-2 to 0-1 and stores a single haplotype dosage.
Unequal haplotype dosages on a reconstructed haploid call, and nonmissing
dosages on female Y, are errors. Diploid mode preserves the native 0-2 scale.
The fixed-point precision in PGEN is exactly representable by the output's
float32 dtype.

For multiallelic data, DS has an `alt_alleles` axis (omitted for biallelic
datasets). HDS is a VCF-style vector with dimension `FORMAT_HDS_dim`:
alternate alleles for haplotype 0, then alternate alleles for haplotype 1.
Each variant's actual allele count determines packing; trailing slots and
absent haplotypes receive VCZ fill NaNs, distinct from missing-data NaNs.
Reference-only variants have only fill slots; an entirely reference-only
dataset omits DS/HDS.

Upstream pgenlib currently supports stored fractional dosages on biallelic
variants. Multiallelic hardcall-derived DS/HDS are supported, but native
multiallelic dosage records raise an explicit backend error, not a fallback
to hardcalls.

Use `--no-dosages` (Python: `dosages=False`) for explicitly hardcall-only
conversion, avoiding the additional decoding, storage and compression work.
This option deliberately omits any stored dosages.

### Parallel encoding

Metadata and file-wide phasedness are read once, without decoding genotypes to
discover output dimensions. Native batch reads release the GIL. Each thread owns
an independent PGEN reader and writes disjoint variant-by-sample chunk tiles.
PGEN uses the shared VCZ writer interface for array initialization, partition
finalization and indexing; thread configuration remains specific to its writer.
Decoded working arrays scale with chunk dimensions, not the entire sample
cohort. Dosages are streamed one alternate allele at a time, with unit-sized
chunks on the DS alternate/HDS vector axes, so highly multiallelic sites do not
multiply working memory by the global allele count.
DS/HDS use bitshuffle before Zstandard compression, matching PGEN's limited
fixed-point precision.
A bounded queue limits outstanding thread tasks.
Alternatively, `--worker-processes` uses the shared process-based encoder;
do not combine it with `--worker-threads`.

PGEN's compression is variant-oriented: reading separate sample tiles can
repeat decompression of the same variant. Larger sample chunks reduce that
overhead, at the cost of larger per-worker buffers. There is no claim that
more threads always improve throughput; storage and compression can dominate.

Phenotype/pedigree columns and additional PVAR annotations are **not** exported.
Directory and ZIP outputs are published only after encoding and indexing
succeed.

:::{note}
Currently we only convert the basic VCF-like data from plink, and don't include
phenotypes and pedigree information. These are planned as future enhancements.
Please comment on [this issue](https://github.com/sgkit-dev/bio2zarr/issues/392)
if you are interested in this functionality.
:::
