#$ delay 10
vcf2zarr convert sample.vcf.gz sample.zarr -p 4
#$ expect \$ 
