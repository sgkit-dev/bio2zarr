# Installation


```
$ python3 -m pip install bio2zarr
```

This will install the programs ``vcf2zarr``, ``plink2zarr`` and ``vcf_partition``
into your local Python path. You may need to update your $PATH to call the
executables directly.

Alternatively, calling
```
$ python3 -m bio2zarr vcf2zarr <args>
```
is equivalent to

```
$ vcf2zarr <args>
```
and will always work.


## Shell completion

To enable shell completion for a particular session in Bash do:

```
eval "$(_VCF2ZARR_COMPLETE=bash_source vcf2zarr)"
```

If you add this to your ``.bashrc`` vcf2zarr shell completion should available
in all new shell sessions.

See the [Click documentation](https://click.palletsprojects.com/en/8.1.x/shell-completion/#enabling-completion)
for instructions on how to enable completion in other shells.
