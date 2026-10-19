(sec-installation)=
# Installation

``bio2zarr`` can either be installed from bioconda or PyPI.

By default the ``bio2zarr`` PyPI package does not come with dependencies
needed to convert VCF, BGEN or tskit files, so optional dependencies
need to be specified:
```bash
python3 -m pip install bio2zarr #Base package with only plink support
python3 -m pip install bio2zarr[vcf] #Install with VCF support
python3 -m pip install bio2zarr[tskit] #Install with tskit support
python3 -m pip install 'bio2zarr[bgen]' #Install with BGEN support
python3 -m pip install bio2zarr[all] #Install with all optional dependencies
```

For bioconda users, a package is available in the
[bioconda channel](https://anaconda.org/bioconda/bio2zarr):
```bash
conda install -c bioconda bio2zarr
```

This will install the programs ``vcf2zarr``, ``vcf_partition``, ``tskit2zarr``
``bgen2zarr`` and ``plink2zarr`` into your local Python path. You may need to update your $PATH to call the
executables directly.

BGEN support uses the **bgen-limix** distribution, version 4.13.0 or later
(below 5), from [jpfeuffer/bgen-limix](https://github.com/jpfeuffer/bgen-limix).
Its import namespace is named `cbgen`, but it is not the legacy `cbgen`
distribution. Do not install the two distributions in the same environment,
because their import namespaces conflict. Python 3.11+ and NumPy 2+ are
required. Backend wheels target macOS ARM64, Linux x86-64/ARM64
and Windows x86-64; other platforms, including Intel macOS, require a source
build with the backend's native build dependencies (C/C++, CMake, zlib and
zstd). Bioconda packaging may lag PyPI's BGEN extra.

Alternatively, calling for example: 
```bash
python3 -m bio2zarr vcf2zarr <args>
```
is equivalent to

```bash
vcf2zarr <args>
```
and will always work.

:::{warning}
Windows support is preliminary, partial and needs to be fully documented.
We recommend trying the bioconda packages in the first instance, and if
this doesn't work try using Windows Subsystem for Linux (WSL).
Please comment on
[this issue](https://github.com/sgkit-dev/bio2zarr/issues/174) if you would
like to see improved Windows support for bio2zarr, or would like to
help out with this.
:::


## Shell completion

To enable shell completion for a particular session in Bash do:

```bash
eval "$(_VCF2ZARR_COMPLETE=bash_source vcf2zarr)"
eval "$(_TSKIT2ZARR_COMPLETE=bash_source tskit2zarr)"
eval "$(_PLINK2ZARR_COMPLETE=bash_source plink2zarr)"
eval "$(_BGEN2ZARR_COMPLETE=bash_source bgen2zarr)"
```

If you add this to your ``.bashrc`` shell completion should available
in all new shell sessions.

See the [Click documentation](https://click.palletsprojects.com/en/8.1.x/shell-completion/#enabling-completion)
for instructions on how to enable completion in other shells.
