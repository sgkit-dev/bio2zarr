#$ delay 5
vcf2zarr explode sample.vcf.gz sample.icf 
#$ expect \$ 
vcf2zarr encode sample.icf sample.vcz
#$ expect \$ 
