---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Bash
  language: bash
  name: bash
---
(sec-vcf2zarr-tutorial)=
# Tutorial

This is a step-by-step tutorial showing you how to convert your
VCF data into Zarr format. There's three different ways to
convert your data, basically providing different levels of
convenience and flexibility corresponding to what you might
need for small, intermediate and large datasets.

:::{warning}
The documentation of vcf2zarr is under development, and 
some bits are more polished than others. This "tutorial"
is experimental, and will likely evolve into a slightly
different format in the near future. It is 
a work in progress and incomplete. The 
{ref}`sec-vcf2zarr-cli-ref` should be complete
and authoritative, however.
:::


## Small dataset

The simplest way to convert VCF data to Zarr is to use the
{ref}`vcf2zarr convert<cmd-vcf2zarr-convert>` command:

:::{tip}
Hit the play button to see the process in action!
:::
<div id="vcf2zarr_convert"></div>
<script>
AsciinemaPlayer.create('../_static/vcf2zarr_convert.cast',
   document.getElementById('vcf2zarr_convert'), {
   cols:80,
   rows:12
});
</script>


This converts the input VCF file to the output ``sample.vcz``
Zarr data in a single pass. The Zarr dataset is stored in the
file system, and you can have a look around using the usual
tools:

```{code-cell}
ls sample.vcz
```

Each of the directories corresponds to an array in Zarr, or
one of the fields in your VCF. The chunk data for each
of these arrays is then stored hierarchically within
these directories:

```{code-cell}
find sample.vcz -maxdepth 2 -type d | sort
```

You can get a better idea of what's being stored and the sizes
of the different fields using the 
{ref}`vcf2zarr inspect<cmd-vcf2zarr-inspect>` command:

```{code-cell}
vcf2zarr inspect sample.vcz
```

The ``stored`` and ``size`` columns here are important, and tell you 
how much storage space is being used by the compressed chunks,
and the full size of the uncompressed array. The ``ratio`` 
column is the ratio of these two values, with large numbers 
indicating a high compression ration. In this example 
the compression ratios are less than one because the compressed
chunks are *larger* than the actual arrays. This is because it's 
a tiny example, with only 9 variants and 3 samples (see the ``shape``
column), so, for example ``call_genotype`` is only 54 bytes.


(sec-vcf2zarr-tutorial-medium-dataset)=
## Medium dataset

Conversion in ``vcf2zarr`` is a two step process. First we convert the VCF(s) to 
an "intermediate columnar format" (ICF), and then use this as the basis of 
generating the final Zarr. 

:::{important}
We would recommend using this two-step process for all but the smallest datasets.
:::

In the simplest case, we just call two commands instead of one:
<div id="vcf2zarr_explode"></div>
<script>
AsciinemaPlayer.create('../_static/vcf2zarr_explode.cast',
   document.getElementById('vcf2zarr_explode'), {
   cols:80,
   rows:12
});
</script>

:::{tip}
Both the {ref}`explode<cmd-vcf2zarr-explode>` 
and
{ref}`encode<cmd-vcf2zarr-encode>` 
commands allow you to perform the
conversion in parallel using the  ``-p/--worker-processes`` option.
:::

The ICF form gives us useful information about the VCF data, and can help us to 
decide some of the finer details of the final Zarr encoding. We can also run
{ref}`vcf2zarr inspect<cmd-vcf2zarr-inspect>` command on an ICF:

```{code-cell}
vcf2zarr inspect sample.icf
```

This tells us about the fields found in the input VCF, and provides details 
about their data type, raw and compressed size, the maximum dimension of a value
and the maximum and minimum values observed in the field. These extreme values
are use to determine the width of integer types in the final Zarr encoding. 
The defaults can be overridden by generating a "storage schema" 
using the {ref}`vcf2zarr mkschema<cmd-vcf2zarr-mkschema>` command on an ICF:

```{code-cell}
vcf2zarr mkschema sample.icf > sample.schema.json
head -n 20 sample.schema.json
```

We've displayed the first 20 lines here so you can get a feel for the JSON format.
The [jq](https://jqlang.github.io/jq/) tool provides a useful way of manipulating
these schemas. Let's look at the schema for just the ``call_genotype``
field, for example:

```{code-cell}
jq '.fields[] | select(.name == "call_genotype")' < sample.schema.json
```

If we wanted to trade-off file size for better decoding performance
we could turn off the shuffle option here (which is set to 2, for 
Blosc BitShuffle). This can be done by loading the ``sample.schema.json``
file into your favourite editor, and changing the value "2" to "0".

:::{todo}
There is a lot going on here. It would be good to give some specific
instructions for how to do this using jq. We would also want to 
explain the numbers provided for shuffle, as well as links to the 
Blosc documentation.
Better mechanisms for updating schemas via a Python API would be 
a useful addition here.
:::


A common thing we might want to do is to drop some fields entirely
from the final Zarr, either because they are too big and unwieldy 
in the standard encoding (e.g. ``call_PL``) or we just don't 
want them.

Suppose here wanted to drop the ``FORMAT/HQ``/``call_HQ`` field. This 
can be done using ``jq`` like:

```{code-cell}
vcf2zarr mkschema sample.icf \
| jq 'del(.fields[] | select(.name == "call_HQ"))' > sample_noHQ.schema.json 
```
Then we can use the updated schema as input to ``encode``:


<!-- FIXME shouldn't need to do this, but currently the execution model is very --> 
<!-- fragile. -->
<!-- https://github.com/sgkit-dev/bio2zarr/issues/238 -->
```{code-cell}
:tags: [remove-cell]
rm -fR sample_noHQ.vcz
```

```{code-cell}
vcf2zarr encode sample.icf -Qs sample_noHQ.schema.json sample_noHQ.vcz
```
:::{tip}
Use the ``-Q/--no-progress`` flag to suppress progress bars.
:::

We can then ``inspect`` to see that there is no ``call_HQ`` array in the output:

```{code-cell}
vcf2zarr inspect sample_noHQ.vcz
```

:::{tip}
Use the ``max-variants-chunks`` option to encode the first few chunks of your 
dataset while doing these kinds of schema tuning operations!
:::

## Large dataset

The {ref}`explode<cmd-vcf2zarr-explode>` 
and {ref}`encode<cmd-vcf2zarr-encode>` commands have powerful features for 
conversion on a single machine, and can take full advantage of large servers
with many cores. Current biobank scale datasets, however, are so large that 
we must go a step further and *distribute* computations over a cluster. 
Vcf2zarr provides some low-level utilities that allow you to do this, that should 
be compatible with any cluster scheduler. 

The distributed commands are split into three phases:

- **init <num_partitions>**: Initialise the computation, setting up the data structures needed
for the bulk computation to be split into ``num_partitions`` independent partitions
- **partition <j>**: perform the computation of partition ``j``
- **finalise**: Complete the full process.

When performing large-scale computations like this on a cluster, errors and job
failures are essentially inevitable, and the commands are resilient to various
failure modes.

Let's go through the example above using the distributed commands. First, we 
{ref}`dexplode-init<cmd-vcf2zarr-dexplode-init>` to create an ICF directory:

```{code-cell}
:tags: [remove-cell]
rm -fR sample-dist.icf
```
```{code-cell}
vcf2zarr dexplode-init sample.vcf.gz sample-dist.icf -n 5 -Q
```

Here we asked ``dexplode-init`` to set up an ICF store in which the data 
is split into 5 partitions. The number of partitions determines the level
of parallelism, so we would usually set this to the number of 
parallel jobs we would like to use. The output of ``dexplode-init`` is 
important though, as it tells us the **actual** number of partitions that 
we have (partitioning is based on the VCF indexes, which have a limited
granularity). You should be careful to use this value in your scripts 
(the format is designed to be machine readable using e.g. ``cut`` and 
``grep``).  In this case there are only 3 possible partitions.


Once ``dexplode-init`` is done and we know how many partitions we have,
we need to call 
{ref}`dexplode-partition<cmd-vcf2zarr-dexplode-partition>` this number of times:

```{code-cell}
vcf2zarr dexplode-partition sample-dist.icf 0
vcf2zarr dexplode-partition sample-dist.icf 1
vcf2zarr dexplode-partition sample-dist.icf 2
```

This is not how it would be done in practise of course: you would 
use your cluster scheduler of choice to dispatch these operations.
:::{todo}
Document how to do this conveniently over some popular schedulers.
:::

:::{tip}
Use the ``--one-based`` argument in cases in which it's more convenient
to index the partitions from 1 to n, rather than 0 to n - 1.
:::

Finally we need to call 
{ref}`dexplode-finalise<cmd-vcf2zarr-dexplode-finalise>`:
```{code-cell}
vcf2zarr dexplode-finalise sample-dist.icf
```

:::{todo}
Document the process for dencode, noting the information output about 
memory requirements.
:::
