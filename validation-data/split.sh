#!/bin/bash
VCF=$1
NUM_PARTS=$2

regions=$( PYTHONPATH="../" python3 -m bio2zarr vcf-partition $VCF -n $NUM_PARTS )
dest=$VCF.$NUM_PARTS.split
echo $dest
rm -fR $dest
mkdir $dest
# TODO figure out how to do this with xargs
for r in $regions; do
	f=$dest/$r.vcf.gz
	bcftools view $VCF -r $r -Oz > $f
	bcftools index $f
done
