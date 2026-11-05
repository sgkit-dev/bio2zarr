# Ploidy-first two-pass VCF benchmark

`vcf_two_pass.py` is an executable, synthetic benchmark, not a production
converter. It compares:

- **ICF:** existing `vcf.explode()` followed by `vcf.encode()`, with default
  16 MiB per-field ICF buffers.
- **Two-pass prototype:** one full streaming scan that infers maximum ploidy,
  record count, allele/FORMAT widths and numeric bounds, followed by one fused
  streaming pass writing all fields. No rows are retained between passes.
  Schema generation, sanitizers, chunk buffers, compressors, finalization and
  indexing come from existing bio2zarr APIs.
- **GT-only scan:** time to decode GT and determine maximum ploidy/count,
  reported separately, **not** substituted for complete conversion.

The optional `--production-direct` additionally measures
`vcf.convert(..., infer_ploidy=True)` on a revision providing that API. It does
not add a redundant GT scan before production conversion.

## Reproduction

From the repository root, using an installed supported Python:

```sh
uv sync --python 3.12 --no-default-groups --extra vcf --group lint
uv run --no-sync python benchmarks/vcf_two_pass.py --result result.json
uv run --no-sync python benchmarks/vcf_two_pass.py \
  --variants 4000 --samples 500 --result gt-4000x500.json
uv run --no-sync python benchmarks/vcf_two_pass.py \
  --variants 2000 --samples 1000 --result gt-2000x1000.json
uv run --no-sync python benchmarks/vcf_two_pass.py \
  --fields GT:DP:PL --result gt-dp-pl-2000x500.json
```

Default dimensions are 2,000 variants, 500 samples, diploid maximum, with three
repeats, variant chunks of 128 and sample chunks of 250. Each repeat uses fresh
worker processes and rotates method order. No downloads or external data are
needed. Each worker uses zero multiprocessing workers, one Blosc compression
thread, and one OpenBLAS/OpenMP thread. Temporary inputs, ICF and outputs are
automatically removed; the requested JSON result persists. The shared-host caps
are 40 million genotype alleles, 20,000 variants, 2,000 samples, ploidy eight,
variant chunks of 256 and five repeats.

The input is a single-contig, biallelic, ordinary gzip-compressed VCF without an
index. Seeded independent random allele calls include phased/unphased calls,
3% all-missing calls, shorter haploid calls and maximum-ploidy calls in the final
record. The optional DP/PL values are random integers with variable PL widths.
This is intentionally narrower than general VCF support: the prototype's fixed
contig/filter encoding only supports this generator's one-contig/PASS workload.
Do not use it to convert arbitrary VCFs.

## Measurements

Measured on 2026-11-05, macOS 26.6 arm64, Python 3.12.15, bio2zarr
`0.2.2.dev36` at base revision `6245ca46a232096e989437675d988f0099225ed0`.
Dependencies: cyvcf2 0.32.1, NumPy 2.4.4, Zarr 3.1.6, numcodecs 0.16.5.
Values below are medians of three runs; MiB means 1,048,576 bytes.
These are prototype measurements, **not** measurements of the separately
implemented production direct-conversion API.

| Fields; variants x samples | Gzip input MiB | ICF wall s | Two-pass wall s | Speedup | ICF peak RSS MiB | Two-pass peak RSS MiB | Temporary ICF MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| GT; 2,000 x 500 | 0.625 | 1.205 | 0.771 | 1.56x | 93.4 | 52.8 | 0.714 |
| GT; 4,000 x 500 | 1.248 | 2.213 | 1.367 | 1.62x | 116.3 | 54.2 | 1.397 |
| GT; 2,000 x 1,000 | 1.223 | 1.814 | 1.104 | 1.64x | 108.0 | 53.8 | 1.408 |
| GT:DP:PL; 2,000 x 500 | 6.305 | 2.762 | 1.372 | 2.01x | 143.6 | 57.0 | 5.422 |

Two-pass temporary ICF usage is exactly zero in every run. This does **not**
mean zero disk usage: final output and the writer's normal output-staging area
still exist.

| Workload, same order as above | GT-only scan s | Combined schema scan s | Final Zarr MiB | ICF sampled peak work MiB | Two-pass sampled peak work MiB |
|---|---:|---:|---:|---:|---:|
| GT; 2,000 x 500 | 0.065 | 0.113 | 0.474 | 1.189 | 0.478 |
| GT; 4,000 x 500 | 0.129 | 0.226 | 0.941 | 2.338 | 0.941 |
| GT; 2,000 x 1,000 | 0.123 | 0.175 | 0.930 | 2.338 | 0.930 |
| GT:DP:PL; 2,000 x 500 | 0.215 | 0.311 | 4.142 | 9.564 | 4.142 |

Raw wall times, in repeat order:

| Workload | ICF seconds | Two-pass seconds | GT-only seconds |
|---|---|---|---|
| GT; 2,000 x 500 | 1.150, 1.205, 1.214 | 0.771, 0.761, 0.776 | 0.067, 0.063, 0.065 |
| GT; 4,000 x 500 | 2.141, 2.324, 2.213 | 1.367, 1.371, 1.363 | 0.131, 0.129, 0.128 |
| GT; 2,000 x 1,000 | 1.814, 1.804, 1.858 | 1.104, 1.068, 1.105 | 0.122, 0.124, 0.123 |
| GT:DP:PL; 2,000 x 500 | 2.646, 2.762, 2.767 | 1.372, 1.368, 1.377 | 0.215, 0.217, 0.214 |

### Metric definitions

Wall time covers schema discovery, conversion, finalization and indexing, but
excludes interpreter/import startup, input generation and correctness checks.
Peak RSS is OS `getrusage(RUSAGE_SELF).ru_maxrss`, including imported libraries;
fresh processes avoid carrying RSS high-water marks across methods. ICF size is
the exact logical size after explosion, when the append-only intermediate is
complete, including metadata/header/indexes.

The parent samples logical work-directory and output-staging bytes at roughly
100 ms intervals. Work bytes include ICF plus all output files, but exclude the
shared input. They are approximate, non-atomic samples: chunk moves during a
directory traversal and brief peaks can affect them. They are not filesystem
allocated bytes, cumulative I/O, or guarantees of exact peak staging usage.
Sampling runs in the parent but may cause minor filesystem contention.

### Correctness and limitations

Every conversion repeat compares the entire array set, shapes, chunk shapes,
dtypes, dimension/description attributes, root attributes and values against the
ICF output. Float comparison uses raw bits, including missing/fill NaN payloads.
Genotype shape and GT-only inferred dimensions must also match the generator.
Region indexes, phasing, missing masks and shorter-call padding are included.
Separate smoke runs covered haploid GT:DP:PL and mixed haploid/triploid GT:DP:PL,
partial final variant chunks and partial sample chunks:

```sh
uv run --no-sync python benchmarks/vcf_two_pass.py \
  --variants 31 --samples 19 --ploidy 1 --fields GT:DP:PL \
  --chunk 16 --sample-chunk 7 --repeats 1
uv run --no-sync python benchmarks/vcf_two_pass.py \
  --variants 33 --samples 12 --ploidy 3 --fields GT:DP:PL \
  --chunk 16 --sample-chunk 5 --repeats 1
uv run --no-sync ruff check benchmarks/vcf_two_pass.py
uv run --no-sync ruff format --check benchmarks/vcf_two_pass.py
```

These small inputs fit in the filesystem cache; caches were not flushed on the
shared host. Gzip is not BGZF/BCF, there is no indexed/multi-contig partitioning,
and no remote/network I/O. High-entropy GT:DP:PL is still not representative of
arbitrary large multiallelic likelihood arrays. A 1 TB ICF or throughput on a
terabyte-scale file cannot be predicted by multiplying these timings or ratios.

The useful architectural result is bounded streaming metadata and output
buffers instead of a materialized genotype intermediate. Eliminating ICF
eliminates that temporary storage term regardless of file size; it does not
eliminate the final Zarr, input storage or the cost of reading the VCF twice.
Across the bounded GT variant-count sweep, prototype RSS remained about
53-54 MiB, while temporary ICF grew with record count. This supports the approach
but does not establish production-scale memory/throughput guarantees.

A GT-only inference API is useful independently, but GT width alone is
insufficient for general schema inference (record count, alleles, Number=G/. and
integer bounds also matter). Combining those discoveries with ploidy inference,
or reusing returned dimension metadata, avoids a third complete parse.
