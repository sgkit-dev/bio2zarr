# Oxford BGEN interoperability fixtures

These are public reference/example datasets, **not verified real-cohort data**.
They were produced independently of bio2zarr and are distributed with
matching text representations for use as independent test oracles.

Source: [OpenMendel/BGEN.jl data directory](https://github.com/OpenMendel/BGEN.jl/tree/7a448d27093125917c2e884303da3a38046db1d6/data),
pinned revision `7a448d27093125917c2e884303da3a38046db1d6`.
Its data-specific `LICENSE.md` attributes the files to the
[Oxford reference implementation](https://enkre.net/cgi-bin/code/bgen/dir?ci=trunk)
and states the Boost Software License 1.0 redistribution terms. The notice is
included unchanged alongside the data.

| BGEN | Companion | Coverage |
| --- | --- | --- |
| `example.8bits.bgen` | `example.first98.gen.gz`, `example.sample` | 500 samples, 199 variants, uncertain diploid probabilities, two missing calls |
| `complex.23bits.bgen` | `complex.vcf` | 4 samples, 10 variants, phased/unphased, variable ploidy and multiple alleles |
| `haplotypes.bgen` | `haplotypes.haps` | 4 samples, 4 variants, ordered phased haplotypes |

All upstream bytes are retained except `example.gen`, whose first 98 lines
are stored as `example.first98.gen.gz` using `gzip -n` (about 395 KiB).
This stays below the repository's 500 KiB per-file hook limit. Reproduce with
`head -n 98 example.gen | gzip -n > example.first98.gen.gz`.
The original complete uncompressed GEN SHA-256 is
`59e5484e558276fcdf6d420c356bec209169774e661d1e82584ab2d7164043de`.
`SHA256SUMS` pins every stored data file and the original license notice.
CI verifies these hashes and requires no downloads.

The full `example` file is intentionally kept unchanged: its position order
falls from 99000 to 10000 at zero-based record 98. The converter requires
nondecreasing positions. A test verifies rejection of the full example;
successful conversion tests select its first 98 records by copying their
original BGEN bytes and changing only the header variant count. No probability
is regenerated, quantized or reordered. The corresponding GEN rows provide the
oracle. GEN and 8-bit BGEN have different quantization; comparisons use explicit
encoding-aware tolerances, not false bit-exact expectations.

The tests compare full probability/haplotype vectors, identifiers and allele
order, ploidy, missing flags and exact VCZ NaN sentinels, and per-variant dosage
totals/frequencies. Complex/HAPS hard calls independently determine expected
per-allele dosages. Summary statistics supplement rather than replace full
array comparisons, since sample/phase permutations can leave aggregates intact.

In `complex.23bits.bgen`, the all-haploid records are marked phased, while the
matching VCF calls their equivalent single-haplotype vectors GP. Tests compare
those vectors to HP and separately check the BGEN phase flags; non-haploid
GP and HP are never conflated.
