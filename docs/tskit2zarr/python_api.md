(sec-tskit2zarr-python-api)=
# Python API

Basic usage:
```python
import bio2zarr.tskit as ts2z

ts2z.convert(ts_path, vcz_path, worker_processes=8)
```

This will convert the [tskit](https://tskit.dev) tree sequence stored
at ``ts_path`` to VCF Zarr stored at ``vcz_path`` using 8 worker processes.
The details of how we map from the
tskit {ref}`tskit:sec_data_model` to VCF Zarr are taken care of by
{meth}`tskit.TreeSequence.map_to_vcf_model`
method, which is called with no
parameters by default if the ``model_mapping`` parameter to
{func}`~bio2zarr.tskit.convert` is not specified.

For more control over the properties of the output, for example
to pick a specific subset of individuals, you can use
{meth}`~tskit.TreeSequence.map_to_vcf_model`
to return the required mapping:

```python
model_mapping = ts.map_to_vcf_model(individuals=[0, 1])
ts2z.convert(ts, vcz_path, model_mapping=model_mapping)
```


## API reference

```{eval-rst}

.. autofunction:: bio2zarr.tskit.convert

```
