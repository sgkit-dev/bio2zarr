(sec-vcf2zarr)=
# vcf2zarr

Convert VCF data to the
[VCF Zarr specification](https://github.com/sgkit-dev/vcf-zarr-spec/)
reliably, in parallel or distributed over a cluster.

See the {ref}`sec-vcf2zarr-tutorial` for a step-by-step introduction
and the {ref}`sec-vcf2zarr-cli-ref` detailed documentation on
command line options.

See the [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2024.06.11.598241) for 
further details.

## Quickstart

- First {ref}`install bio2zarr<sec-installation>`.


- Get some indexed VCF data:

```
curl -O https://raw.githubusercontent.com/sgkit-dev/bio2zarr/main/tests/data/vcf/sample.vcf.gz
curl -O https://raw.githubusercontent.com/sgkit-dev/bio2zarr/main/tests/data/vcf/sample.vcf.gz.tbi
```

- Convert to VCF Zarr in two steps:

```
vcf2zarr explode sample.vcf.gz sample.icf
vcf2zarr encode sample.icf sample.vcz
```

:::{tip}
If the ``vcf2zarr`` executable doesn't work, try ``python -m bio2zarr vcf2zarr``
instead.
:::

- Have a look at the results:

```
vcf2zarr inspect sample.vcz
```

### What next?

VCF Zarr is a starting point in what we hope will become a diverse ecosytem
of packages that efficiently process VCF data in Zarr format. This 
ecosytem is in its infancy and there isn't much software available
for performing off-the-shelf bioinformatics tasks
working with the format. As such, VCF Zarr isn't suitable for end users
who just want to get their work done for the moment, and is currently
aimed methods developers and early adopters.

Having said that, you can:

- Use [vcztools](https://github.com/sgkit-dev/vcztools/) as a drop-in replacment 
  for bcftools, transparently using Zarr on local storage or cloud stores as the 
  backend.
- Look at the [VCF Zarr specification](https://github.com/sgkit-dev/vcf-zarr-spec/)
  to see how data is mapped from VCF to Zarr
- Use the mature [Zarr Python](https://zarr.readthedocs.io/en/stable/) package or
one of the other [Zarr implementations](https://zarr.dev/implementations/) to access
your data.
- Use the many functions in our [sgkit](https://sgkit-dev.github.io/sgkit/latest/)
sister project to analyse the data. Note that sgkit is under active development,
however, and the documentation may not be fully in-sync with this project.

For more information, please see our 
bioRxiv preprint [Analysis-ready VCF at Biobank scale using Zarr](
https://www.biorxiv.org/content/10.1101/2024.06.11.598241).


## How does it work?
The conversion of VCF data to Zarr is a two-step process:

1. Convert ({ref}`explode<cmd-vcf2zarr-explode>`) VCF file(s) to
    Intermediate Columnar Format (ICF)
2. Convert ({ref}`encode<cmd-vcf2zarr-encode>`) ICF to Zarr

This two-step process allows `vcf2zarr` to determine the correct
dimension of Zarr arrays corresponding to each VCF field, and
to keep memory usage tightly bounded while writing the arrays.

:::{important}
The intermediate columnar format is not intended for any use
other than a temporary storage while converting VCF to Zarr.
The format may change between versions of `bio2zarr`.
:::

Both ``explode`` and ``encode`` can be performed in parallel
across cores on a single machine (via the ``--worker-processes`` argument)
or distributed across a cluster by the three-part ``init``, ``partition``
and ``finalise`` commands.

## Batched conversion without ICF

`convert` uses row batches sized to the output variant chunks by default
(normally 1000 rows). With a known maximum
ploidy, it can avoid writing ICF entirely:

```bash
vcf2zarr convert sample.vcf.gz sample.vcz --max-ploidy 2
vcf2zarr convert sample.vcf.gz sample.vcz --infer-shape
vcf2zarr convert sample.vcf.gz sample.vcz --infer-shape --max-ploidy 2 --max-alleles 4
```

Maximum ploidy alone does not bound allele counts, variable-width INFO/FORMAT
fields, or integer values. The no-ICF path therefore makes a streaming metadata
pass to discover these dimensions and dtypes, keeping only summaries, then
reads records again and converts all fields from shared row batches. With
`--max-ploidy`, metadata discovery does not decode GT; the bound is checked
during conversion. `--infer-shape` discovers all array dimensions and dtypes,
including GT width when it is not supplied. It may accompany either or both
maximum bounds, inferring the remaining dimensions. Neither path writes
genotype or other field ICF data.

`--max-alleles` includes REF: 2 is biallelic, and 1 permits only monomorphic
records. The supplied bound sizes `alleles` and `alt_alleles`; with known or
inferred ploidy it also determines the combinatorial width of Number=G fields.
Too many actual alleles, or field values exceeding these widths, raise errors
rather than truncating data. `--max-alleles` alone selects no-ICF conversion
with ploidy inferred during the metadata pass. Other variable-width fields
and numeric dtypes are still discovered: even supplying both maximum bounds
does **not** make this a single-pass conversion.

`--batch-size` limits rows; `--max-batch-bytes` limits approximate batch payload
(default 64 MiB per worker, including string contents). Batches also stop at
output variant-chunk boundaries, so narrow rows normally write each chunk once.
Decoder objects, Zarr codec/chunk buffers and
Python overhead add to this payload; this is not a process RSS limit. A single
row exceeding the budget raises an error rather than being truncated. Larger
rows require an explicitly increased budget. Ploidy, allele-count, field-width
and integer-dtype overflows raise errors.

The existing `--ploidy` option pads the conventional ICF-based conversion.
It may also pad `--infer-shape` / `--max-alleles` conversion, but is mutually
exclusive with `--max-ploidy`. Local alleles, missing/fill
sentinels, phasing, zip output and Zarr 2/3 remain available. For explicit
ICF workflows, `encode --batch-size N` batches source fields one column at a
time (rather than retaining a decompressed ICF chunk for every field).
`dencode-init --batch-size N` records this choice for subsequent workers;
the existing distributed lifecycle is unchanged.

## S3 input and intermediate storage

Install `bio2zarr[vcf,s3]` to use `s3://bucket/key` VCF/BCF input and
`s3://bucket/prefix` ICF storage:

```bash
vcf2zarr convert s3://my-bucket/input/sample.vcf.gz sample.vcz --max-ploidy 2
vcf2zarr explode s3://my-bucket/input/sample.vcf.gz s3://my-bucket/work/sample.icf
vcf2zarr encode s3://my-bucket/work/sample.icf sample.vcz --batch-size 64
vcf2zarr convert sample.vcf.gz sample.vcz --temp-icf-dir s3://my-bucket/tmp
```

S3 VCF data is read by htslib using signed HTTP URLs with range requests;
only small adjacent `.tbi`/`.csi` indexes are downloaded to a temporary local
directory. Indexed multi-contig input and local input have the same partition
semantics. The cyvcf2/htslib build must support HTTP. URLs are freshly signed
for each opened partition, valid for up to 24 hours (or the remaining credential
lifetime, if shorter). Very long individual partition reads require smaller
partitions or longer-lived credentials. Native HTTP/authentication errors
propagate rather than being replaced by empty results.

Credentials come from the standard AWS credential chain (environment, profiles,
or IAM roles). `AWS_ENDPOINT_URL_S3` or `AWS_ENDPOINT_URL` selects an
S3-compatible endpoint. Every distributed worker needs the same credentials
and endpoint configuration. Credentials and signed URLs are not written to
ICF or conversion metadata. Buckets must already exist. Use a dedicated,
otherwise-empty prefix for each ICF run; a prefix containing existing data is
rejected.

`--icf-path` retains intermediate data; `--temp-icf-dir` chooses a local parent
directory or S3 parent prefix for temporary ICF. The latter creates a unique
run prefix and removes only that prefix on success or failure. These options
are mutually exclusive, and neither may be used with no-ICF conversion.
`explode`, `inspect`, `mkschema`, `encode` and all distributed ICF commands
accept S3 ICF prefixes. Final VCZ output remains local (directory or zip);
S3 VCZ output is not supported by this converter.

## Local alleles

As discussed in our [preprint](
https://www.biorxiv.org/content/10.1101/2024.06.11.598241) 
vcf2zarr has an experimental implementation of the local alleles data
reduction technique. This essentially reduces the inner dimension of 
large fields such as AD by storing information relevant only to the alleles
involved in a particular variant call, rather than information information
for all alleles. This can make a substantial difference when there is a large 
number of alleles.

To use local alleles, you must generate storage a schema (see the 
{ref}`sec-vcf2zarr-tutorial-medium-dataset` section of the tutorial)
using the {ref}`mkschema<cmd-vcf2zarr-mkschema>` command with the 
``--local-alleles`` option. This will generate the ``call_LA`` field
which lists the alleles observed for each genotype call, and 
translate supported fields from their global alleles to local
alleles representation.

:::{warning}
Support for local-alleles is preliminary and may be subject to change
as the details of how alleles for a particular call are chosen, and the 
number of alleles retained determined. Please open an issue on
[GitHub](https://github.com/sgkit-dev/bio2zarr/issues/) if you would like to 
help improve Bio2zarr's local alleles implementation.
:::

:::{note}
Only the PL and AD fields are currently supported for local alleles
data reduction. Please comment on our 
[local alleles fields tracking issue](https://github.com/sgkit-dev/bio2zarr/issues/315)
if you would like to see other fields supported, or to help out with 
implementing more.
:::

## Debugging

When things go wrong with conversion, it's very useful to generate some 
debugging information using the ``-v`` or ``-vv`` options. These can 
help identify what's going wrong (usually running out of memory).

:::{warning}
To get the full logging output you **must use** -p0, such that no multiprocessing
is used. This means that tracking down problems can be slow, unfortunately.
This is due to a bug in the way logging from child processes is handled;
see [issue 302](https://github.com/sgkit-dev/bio2zarr/issues/302) for details.
:::



## Copying to object stores

:::{todo}
Document process of copying VCF Zarr datasets to an object store like S3.
See [Issue 234](https://github.com/sgkit-dev/bio2zarr/issues/234)
:::
