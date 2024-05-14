# Installation


```bash
python3 -m pip install bio2zarr
```

This will install the programs ``vcf2zarr`` and ``vcf_partition``
into your local Python path. You may need to update your $PATH to call the
executables directly.

Alternatively, calling
```bash
python3 -m bio2zarr vcf2zarr <args>
```
is equivalent to

```bash
vcf2zarr <args>
```
and will always work.


:::{warning}
Windows is not currently supported. Please comment on
[this issue](https://github.com/sgkit-dev/bio2zarr/issues/174) if you would
like to see Windows support for bio2zarr.
:::


## Shell completion

To enable shell completion for a particular session in Bash do:

```bash
eval "$(_VCF2ZARR_COMPLETE=bash_source vcf2zarr)"
```

If you add this to your ``.bashrc`` vcf2zarr shell completion should available
in all new shell sessions.

See the [Click documentation](https://click.palletsprojects.com/en/8.1.x/shell-completion/#enabling-completion)
for instructions on how to enable completion in other shells.
