# 0.1.5 2025-03-xx

- Add support for merging contig IDs across multiple VCFs (#335)

- Add support for unindexed (and uncompressed) VCFs (#337)

# 0.1.4 2025-03-10

- Fix bug in handling all-missing genotypes (#328)

# 0.1.3 2025-03-04

- Fix missing dependency issue for packaging

- Support out-of-order field definitions in the VCF header (#322, @ACEnglish)

# 0.1.2 2025-02-04

- Reduce memory requirement for encoding genotypes with large sample sizes

- Transpose default chunk sizes to 1000 variants and 10,000 samples (issue:300)

- Add chunksize options to mkschema (issue:294)

- Add experimental support for local alleles.

- Add experimental support for ``region_index``

Breaking changes

- ICF metadata format version bumped to ensure long-term compatility between numpy 1.26.x
  and numpy >= 2. Existing ICFs will need to be recreated.


# 0.1.1 2024-06-19

Maintenance release:

- Pin numpy to < 2
- Pin Zarr to < 3

# 0.1.0 2024-06-10

- Initial production-ready version.
- Add -Q/--no-progress flag to CLI
- Change num-partitions argument in dexplode-init and dencode-init
  to a named option.

# 0.0.10 2024-05-15
- Change output format of dexplode-init and dencode-init
- Bugfix for mac progress, and change of multiprocessing startup strategy.

# 0.0.9 2024-05-02

- Change on-disk format for explode and schema
- Support older tabix indexes
- Fix some bugs in explode

# 0.0.8 2024-04-30

- Change on-disk format of distributed encode and simplify
- Check for all partitions nominally completed encoding before doing
  anything destructive in dencode-finalise

# 0.0.6 2024-04-24

- Only use NOSHUFFLE by default on ``call_genotype`` and bool arrays.
- Add initial implementation of distributed encode

# 0.0.5 2024-04-17

- Fix bug in schema handling (compressor settings ignored)
- Move making ICF field partition directories into per-partition processing.
  Remove progress on the init mkdirs step.
- Turn off progress monitor on dexplode-partition
- Fix empty partition bug

# 0.0.4 2024-04-08

- Fix bug in --max-memory handling, and argument to a string like 10G
- Add compressor choice in explode, switch default to zstd
- Run mkdirs in parallel and provide progress
- Change dimension separator to "/" in Zarr
- Update min Zarr version to 2.17

# 0.0.3 2024-03-28

- Various refinements to the CLI

# 0.0.2 2024-03-27

- Merged 1D and 2D encode steps into one, and change rate reporting to bytes
- Add --max-memory for encode
- Change `chunk_width` to `samples_chunk_size` and `chunk_length` to `variants_chunk_size`
- Various updates to the intermediate chunked format, with breaking change to version 0.2
- Add distributed explode commands
