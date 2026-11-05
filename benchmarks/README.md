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

`--large` explicitly enables up to one million variants, 100 million genotype
alleles, variant chunks of 4,096 and two repeats. It requires at least 8 GiB free
temporary disk. `--info-bytes N` adds a seeded, independently random,
62-character-alphabet INFO string of length N to every row (maximum 2,048).
The generator now records exact uncompressed UTF-8 bytes alongside gzip bytes
and generation time. Output/ICF directories are removed after each repeat's
correctness comparison, rather than accumulating across repeats. The JSON
includes a script checksum for subsequent runs.

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

## Expanded hundreds-of-MB runs

The larger follow-up used the same hardware/software, base benchmark commit
`122f30c` plus the generator extensions above. These are **single-run prototype
measurements**, not statistical medians or production API measurements. The
cases ran serially, with ICF first, prototype second, then the GT-only scan.
No simultaneous conversions or live cloud traffic were used. About 38 GiB
temporary disk was available before and after the runs.

```sh
uv run --no-sync python benchmarks/vcf_two_pass.py --large \
  --variants 40000 --samples 1000 --fields GT:DP:PL \
  --chunk 1024 --sample-chunk 250 --repeats 1 \
  --result large-gt-dp-pl-40000x1000.json
uv run --no-sync python benchmarks/vcf_two_pass.py --large \
  --variants 300000 --samples 4 --info-bytes 512 \
  --chunk 4096 --sample-chunk 4 --repeats 1 \
  --result large-row-info-300000x4.json
uv run --no-sync python benchmarks/vcf_two_pass.py --large \
  --variants 500000 --samples 4 \
  --chunk 4096 --sample-chunk 4 --repeats 1 \
  --result row-low-payload-500000x4.json
```

| Case | Variants x samples | Actual gzip input bytes | Actual raw input bytes | Generation s |
|---|---:|---:|---:|---:|
| Genotype-heavy GT:DP:PL | 40,000 x 1,000 | 263,847,034 | 698,660,307 | 140.06 |
| Row/INFO-heavy GT + 512-byte random INFO | 300,000 x 4 | 121,117,507 | 170,124,047 | 12.46 |
| Low-payload row-heavy GT control | 500,000 x 4 | 4,408,101 | 24,177,378 | 8.50 |

The main inputs are **263.8 MB and 121.1 MB compressed**, not merely hundreds of
MB raw. The low-payload control is intentionally much smaller on disk: it tests
per-record overhead when genotype data is cheap, rather than meeting the size
target. Gzip/raw size ratios are 0.378, 0.712 and 0.182 respectively. Generation
is excluded from all conversion timings.

| Case | ICF wall s | Two-pass wall s | Speedup | ICF peak RSS MiB | Two-pass peak RSS MiB | GT-only s | Combined schema scan s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Genotype-heavy | 86.38 | 29.36 | 2.94x | 682.5 | 81.6 | 8.59 | 11.49 |
| Row/INFO-heavy | 82.80 | 22.34 | 3.71x | 1,009.4 | 257.8 | 1.33 | 8.57 |
| Low-payload row-heavy | 43.62 | 25.97 | 1.68x | 644.5 | 53.8 | 1.13 | 12.57 |

| Case | Temporary ICF bytes | Final Zarr bytes | ICF sampled peak work bytes | Two-pass sampled peak work bytes |
|---|---:|---:|---:|---:|
| Genotype-heavy | 225,575,718 | 172,520,737 | 398,096,455 | 172,520,737 |
| Row/INFO-heavy | 154,679,552 | 116,519,572 | 271,199,124 | 146,866,618 |
| Low-payload row-heavy | 3,674,036 | 1,487,972 | 5,162,008 | 1,487,972 |

Two-pass ICF remains zero. The gzip input is additional to the work-directory
usage in this table. The largest sampled total including input was about
662 MB logical bytes. Non-atomic directory sampling is especially visible in
the INFO case; the sampled work size is not an exact filesystem high-water mark.
Peak RSS includes codec/serialization/native allocations, not just buffers.
The ICF row/INFO case reached about **1 GiB RSS**, exceeding the initial pilot
estimate of 200-500 MiB; serial execution kept this bounded. Prototype
`buffer_bytes` (12,349,440 / 393,216 / 327,680) counts NumPy containers only:
variable-length strings and native codec allocations are not represented.

Every output comparison passed, including the random `variant_PAYLOAD` INFO
values, GT masks/padding/phasing, schema and region index. These observations
remain cache-sized relative to host memory. Input generation warms the
filesystem cache, and method ordering can favor the later run; **cold-cache
performance was not measured**. No system cache flushing or memory-pressure
eviction was attempted on the shared host.

### Where can two-pass be worst?

**Measured:** the low-payload, many-row control gives the weakest speedup,
1.68x. Its first pass consumes about **48%** of two-pass conversion time,
versus 39% for the genotype-heavy case and 38% for the INFO-heavy case.
The GT-only scan is only 1.13 s, while complete schema discovery takes 12.57 s:
cheap GT does not make full per-record metadata work free. This is reduced
benefit, **not an observed regression against ICF**. INFO-heavy input raises
schema work relative to a GT-only scan (8.57 versus 1.33 s), but eliminating
ICF serialization still dominates in this measured case.

**Analytic, not measured here:**

- A remote or cold, slow source must be read twice without input caching. At
  bandwidth B, an additional complete traversal transfers compressed input
  size C again, with a transfer budget C/B; overlap with parsing, remote caching
  and latency determine the actual wall-time penalty. For illustration,
  263.8 MB at 10 MB/s represents 26.4 s of transfer per traversal. This is
  arithmetic, not an S3 measurement. An ICF route parses the VCF once but
  writes and rereads its intermediate; relative source/intermediate bandwidth
  matters. Slow S3 plus fast local intermediate storage can favor ICF.
- Poorly compressible, metadata-heavy inputs can make repeat decompression and
  schema parsing more expensive than rereading a compact local ICF. Our random
  INFO case exercises low compressibility, but does not demonstrate that
  crossover. Highly compressible sources may instead make extra source I/O
  cheap while ICF object serialization still dominates.
- Many rows with very few samples expose fixed Python/header-field processing
  costs. Extra INFO fields, complex Number=./G values or large multiallelic
  likelihoods can make inference more expensive; one random scalar INFO field
  is not coverage of all those cases.
- Repeated indexed partition scans/seeks, shard fan-out, remote retries and
  non-seekable inputs can worsen a design that silently rereads or refetches
  data. The prototype performs exactly two full sequential parses plus the
  existing header peek; it does not test those production behaviors.

The larger results support removing the materialized ICF for these cases.
They do not establish production-API throughput, cold/S3 performance or
terabyte-scale extrapolation.
