# 0.0.5 2024-04-XX

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
