# FIXME this is all very fragile and needs to be rewritten.
# https://github.com/sgkit-dev/bio2zarr/issues/238
CASTS=_static/vcf2zarr_convert.cast\
	_static/vcf2zarr_explode.cast

ASCIINEMA_ARGS=-c "env PS1=\\$$\\  bash --noprofile --norc"

BUILDDIR      = _build

all: ${CASTS}
	./build.sh

clean:
	rm -fR $(BUILDDIR)
	rm -f _static/*.cast
	rm -f sample.vcf.gz*


sample.vcf.gz:
	cp ../tests/data/vcf/sample.vcf.gz ./
	cp ../tests/data/vcf/sample.vcf.gz.tbi ./
	# FIXME we should really running the casts out of the
	# vcf2zarr directory, but let's get this working for now.
	cp sample.vcf.gz* vcf2zarr

_static/vcf2zarr_convert.cast: sample.vcf.gz
	rm -fR sample.vcz
	uv run --project=.. --group docs --no-default-groups asciinema-automation -aa '$(ASCIINEMA_ARGS)' -d cast_scripts/vcf2zarr_convert.sh $@
	cat _static/vcf2zarr_convert.log
	uv run --project=.. --group docs --no-default-groups asciinema play _static/vcf2zarr_convert.cast
	cp -R sample.vcz vcf2zarr

# TODO rename this cast
_static/vcf2zarr_explode.cast: sample.vcf.gz
	rm -Rf sample.icf sample.vcz
	uv run --project=.. --group docs --no-default-groups asciinema-automation -aa '$(ASCIINEMA_ARGS)' -d cast_scripts/vcf2zarr_explode.sh $@
	cat _static/vcf2zarr_explode.log
	uv run --project=.. --group docs --no-default-groups asciinema play _static/vcf2zarr_explode.cast
	cp -R sample.icf sample.vcz vcf2zarr
