# Need to set PYTHONPATH so that we pick up the local bio2zarr
PYPATH=$(shell pwd)/../
B2Z_VERSION:=$(shell PYTHONPATH=${PYPATH} \
   python3 -c 'import bio2zarr; print(bio2zarr.__version__.split("+")[0])')


# FIXME this is all very fragile and needs to be rewritten. 
# https://github.com/sgkit-dev/bio2zarr/issues/238
CASTS=_static/vcf2zarr_convert.cast\
	_static/vcf2zarr_explode.cast


BUILDDIR      = _build

dev: ${CASTS}
	PYTHONPATH=${PYPATH} ./build.sh

dist: ${CASTS}
	@echo Building distribution for bio2zarr version ${B2Z_VERSION}
	cd doxygen && doxygen
	sed -i -e s/__BIO2ZARR_VERSION__/${B2Z_VERSION}/g _config.yml
	PYTHONPATH=${PYPATH} ./build.sh

clean:
	rm -fR $(BUILDDIR)
	rm -f _static/*.cast
	rm sample.vcf.gz*


sample.vcf.gz:
	cp ../tests/data/vcf/sample.vcf.gz ./
	cp ../tests/data/vcf/sample.vcf.gz.tbi ./
	# FIXME we should really running the casts out of the
	# vcf2zarr directory, but let's get this working for now.
	cp sample.vcf.gz* vcf2zarr

_static/vcf2zarr_convert.cast: sample.vcf.gz
	rm -fR sample.vcz
	asciinema-automation -d cast_scripts/vcf2zarr_convert.sh $@
	cat _static/vcf2zarr_convert.log
	asciinema play _static/vcf2zarr_convert.cast 
	cp -R sample.vcz vcf2zarr

# TODO rename this cast
_static/vcf2zarr_explode.cast: sample.vcf.gz
	rm -Rf sample.icf sample.vcz
	asciinema-automation -d cast_scripts/vcf2zarr_explode.sh $@
	cat _static/vcf2zarr_explode.log
	asciinema play _static/vcf2zarr_explode.cast 
	cp -R sample.icf sample.vcz vcf2zarr
