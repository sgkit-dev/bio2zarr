# Need to set PYTHONPATH so that we pick up the local bio2zarr
PYPATH=$(shell pwd)/../
B2Z_VERSION:=$(shell PYTHONPATH=${PYPATH} \
   python3 -c 'import bio2zarr; print(bio2zarr.__version__.split("+")[0])')

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


sample.vcf.gz:
	cp ../tests/data/vcf/sample.vcf.gz ./
	cp ../tests/data/vcf/sample.vcf.gz.tbi ./

_static/vcf2zarr_convert.cast: sample.vcf.gz
	rm -f sample.vcz
	asciinema-automation cast_scripts/vcf2zarr_convert.sh $@

_static/vcf2zarr_explode.cast: sample.vcf.gz
	rm -Rf sample.icf
	asciinema-automation cast_scripts/vcf2zarr_explode.sh $@

