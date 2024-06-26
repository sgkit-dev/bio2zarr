(sec-vcf2zarr-cli-ref)=
# CLI Reference

% A note on cross references... There's some weird long-standing problem with
% cross referencing program values in Sphinx, which means that we can't use
% the built-in labels generated by sphinx-click. We can make our own explicit
% targets, but these have to have slightly weird names to avoid conflicting
% with what sphinx-click is doing. So, hence the cmd- prefix.
% Based on: https://github.com/skypilot-org/skypilot/pull/2834

```{eval-rst}

.. _cmd-vcf2zarr-convert:
.. click:: bio2zarr.cli:convert_vcf
   :prog: vcf2zarr convert
   :nested: full

.. _cmd-vcf2zarr-inspect:
.. click:: bio2zarr.cli:inspect
   :prog: vcf2zarr inspect
   :nested: full

.. _cmd-vcf2zarr-mkschema:
.. click:: bio2zarr.cli:mkschema
   :prog: vcf2zarr mkschema
   :nested: full
```

## Explode

```{eval-rst}
.. _cmd-vcf2zarr-explode:
.. click:: bio2zarr.cli:explode
   :prog: vcf2zarr explode
   :nested: full

.. _cmd-vcf2zarr-dexplode-init:
.. click:: bio2zarr.cli:dexplode_init
   :prog: vcf2zarr dexplode-init
   :nested: full

.. _cmd-vcf2zarr-dexplode-partition:
.. click:: bio2zarr.cli:dexplode_partition
   :prog: vcf2zarr dexplode-partition
   :nested: full

.. _cmd-vcf2zarr-dexplode-finalise:
.. click:: bio2zarr.cli:dexplode_finalise
   :prog: vcf2zarr dexplode-finalise
   :nested: full
```

## Encode

```{eval-rst}
.. _cmd-vcf2zarr-encode:
.. click:: bio2zarr.cli:encode
   :prog: vcf2zarr encode
   :nested: full

.. _cmd-vcf2zarr-dencode-init:
.. click:: bio2zarr.cli:dencode_init
   :prog: vcf2zarr dencode-init
   :nested: full

.. _cmd-vcf2zarr-dencode-partition:
.. click:: bio2zarr.cli:dencode_partition
   :prog: vcf2zarr dencode-partition
   :nested: full

.. _cmd-vcf2zarr-dencode-finalise:
.. click:: bio2zarr.cli:dencode_finalise
   :prog: vcf2zarr dencode-finalise
   :nested: full
```

