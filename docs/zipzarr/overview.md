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
(sec-zipzarr)=
# zipzarr

## Overview

The {ref}`cmd-zipzarr` utility packs a Zarr directory store into a single
``.zip`` file, and can reverse the operation with ``-u/--unzip``.

Output paths default to SRC with ``.zip`` appended (or stripped, under
``-u``), and the source is removed on success unless ``-k/--keep`` is
passed — mirroring ``gzip``.

:::{warning}
``zipzarr`` does **not** check whether the input is a Zarr store. It
will zip any directory and unzip any archive; it is your responsibility
to ensure the input is a well-formed Zarr store.
:::

### Zipping a directory store

Given a Zarr directory store ``sample.vcz``, pack it into a single file:

```
zipzarr sample.vcz sample.vcz.zip
```

If you omit the output path, ``zipzarr`` appends ``.zip`` to the source
and removes the source directory after zipping. Pass ``-k/--keep`` to keep
it:

```
zipzarr sample.vcz        # writes sample.vcz.zip, removes sample.vcz
zipzarr -k sample.vcz     # writes sample.vcz.zip, keeps sample.vcz
```

### Unzipping a zipped store

To go the other way, pass ``-u/--unzip`` and swap the argument order so
that the archive comes first and the target directory second:

```
zipzarr -u sample.vcz.zip sample-roundtrip.vcz
```

If you omit the output path, ``zipzarr`` strips the trailing ``.zip`` and
removes the zip after extracting. Pass ``-k/--keep`` to keep it:

```
zipzarr -u sample.vcz.zip       # writes sample.vcz, removes the .zip
zipzarr -u -k sample.vcz.zip    # writes sample.vcz, keeps the .zip
```

If SRC does not end in ``.zip``, you must provide the output path
explicitly.

### Options

- ``--progress/--no-progress`` shows or hides a progress bar. Progress is
  on by default.
- ``--force`` overwrites an existing destination. Without it, ``zipzarr``
  refuses to clobber an existing file or directory.
- ``-k/--keep`` preserves the source after a successful operation.
  Without it, SRC is removed on success (like ``gzip``).
