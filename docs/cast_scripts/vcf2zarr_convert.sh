#$ delay 10
vcf2zarr convert tests/data/vcf/sample.bcf tmp.zarr -p 4
#$ expect \$ 
