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

