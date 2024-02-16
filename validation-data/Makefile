all: 1kg_2020_chr20.bcf.csi \
	1kg_2020_chr20.vcf.gz.tbi \
	1kg_2020_chr20_annotations.vcf.gz.tbi \
	1kg_2020_chr20_annotations.bcf.csi

.PRECIOUS: %.bcf
.PRECIOUS: %.vcf.gz

1KG_2020_GENOTYPES_URL="http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20201028_3202_raw_GT_with_annot/20201028_CCDG_14151_B01_GRM_WGS_2020-08-05_chr20.recalibrated_variants.vcf.gz"
1KG_2020_ANNOTATIONS_URL="http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20201028_3202_raw_GT_with_annot/20201028_CCDG_14151_B01_GRM_WGS_2020-08-05_chr20.recalibrated_variants.annotated.vcf.gz"


%.vcf.gz.tbi: %.vcf.gz
	tabix $<

%.bcf.csi: %.bcf
	bcftools index $<

%.bcf: %.vcf.gz
	bcftools view -O b $< > $@

1kg_2020_chr20.vcf.gz:
	bcftools view ${1KG_2020_GENOTYPES_URL} | head -n 10000 | bcftools view -O z -o $@

1kg_2020_chr20_annotations.vcf.gz:
	bcftools view ${1KG_2020_ANNOTATIONS_URL} | head -n 100000 | bcftools view -O z -o $@

clean:
	rm -f *.vcf.gz *.tbi *.bcf *.csi
