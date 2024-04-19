# Need to set PYTHONPATH so that we pick up the local bio2zarr
PYPATH=$(shell pwd)/../
B2Z_VERSION:=$(shell PYTHONPATH=${PYPATH} \
   python3 -c 'import bio2zarr; print(bio2zarr.__version__.split("+")[0])')

BUILDDIR      = _build

dev:
	PYTHONPATH=${PYPATH} ./build.sh

dist:
	@echo Building distribution for bio2zarr version ${B2Z_VERSION}
	cd doxygen && doxygen
	sed -i -e s/__BIO2ZARR_VERSION__/${B2Z_VERSION}/g _config.yml
	PYTHONPATH=${PYPATH} ./build.sh

clean:
	rm -fR $(BUILDDIR)
