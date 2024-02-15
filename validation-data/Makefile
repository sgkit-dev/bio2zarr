all: 1kg_2020_chr20.bcf.csi 1kg_2020_chr20.vcf.gz.tbi 

%.vcf.gz.tbi: %.vcf.gz
	tabix $<

%.bcf.csi: %.bcf
	bcftools index $<

.PRECIOUS: %.bcf
%.bcf: %.vcf.gz
	bcftools view -O b $< > $@

.PRECIOUS: %.vcf.gz
1kg_2020_chr20.vcf.gz:
	# TODO replace with URL
	bcftools view /home/jk/work/github/sgkit/big-data/20201028_CCDG_14151_B01_GRM_WGS_2020-08-05_chr20.recalibrated_variants.vcf.gz | head -n 10000 | bcftools view -O z > $@

clean:
	rm -f *.vcf.gz *.tbi *.bcf *.csi
