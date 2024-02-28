all: 1kg_2020_chr20.bcf.csi \
	1kg_2020_chr20.vcf.gz.tbi \
	1kg_2020_chr20.vcf.gz.csi \
	1kg_2020_chr20_annotations.vcf.gz.tbi \
	1kg_2020_chr20_annotations.vcf.gz.csi \
	1kg_2020_chr20_annotations.bcf.csi\
	1kg_2020_others.bcf.csi \
	1kg_2020_others.vcf.gz.tbi \
	1kg_2020_others.vcf.gz.csi \
	1kg_2020_freebayes.vcf.gz.tbi \
	1kg_2020_freebayes.vcf.gz.csi \
	1kg_2020_freebayes.bcf.csi \
	1kg_p3_all_chr1.bcf.csi \
	1kg_p3_all_chr1.vcf.gz.tbi\
	1kg_p3_all_chr1.vcf.gz.csi\
	1kg_p1_all_chr6.bcf.csi\
	1kg_p1_all_chr6.vcf.gz.tbi \
	1kg_p1_all_chr6.vcf.gz.csi

.PRECIOUS: %.bcf
.PRECIOUS: %.vcf.gz

# Modern 1000 genomes data
1KG_2020_BASE=http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20201028_3202_raw_GT_with_annot/20201028_CCDG_14151_B01_GRM_WGS_2020-08-05_
1KG_2020_GENOTYPES_URL="${1KG_2020_BASE}chr20.recalibrated_variants.vcf.gz"
1KG_2020_ANNOTATIONS_URL="${1KG_2020_BASE}chr20.recalibrated_variants.annotated.vcf.gz"
1KG_2020_OTHERS_URL="${1KG_2020_BASE}others.recalibrated_variants.vcf.gz"
1KG_2020_FREEBAYES="http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20200515_EBI_Freebayescalls/ALL.chrX.freebayes.20200518.snps_indels.NYhc.GRCh38.vcf.gz"

# 1000 genomes phase 3
1KG_P3_ALL_URL=http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/release/20190312_biallelic_SNV_and_INDEL/ALL.chr1.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz
1KG_P3_SNP_URL=http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/release/20181203_biallelic_SNV/ALL.chr1.shapeit2_integrated_v1a.GRCh38.20181129.phased.vcf.gz

# 1000 genomes phase 1
1KG_P1_ALL_URL=http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase1/analysis_results/integrated_call_sets/ALL.chr6.integrated_phase1_v3.20101123.snps_indels_svs.genotypes.vcf.gz


%.vcf.gz.tbi: %.vcf.gz
	tabix $<

%.csi: %
	bcftools index $<

%.bcf: %.vcf.gz
	bcftools view -O b $< > $@

1kg_2020_chr20.vcf.gz:
	bcftools view ${1KG_2020_GENOTYPES_URL} | head -n 10000 | bcftools view -O z -o $@

1kg_2020_others.vcf.gz:
	bcftools view ${1KG_2020_OTHERS_URL} | head -n 10000 | bcftools view -O z -o $@

1kg_2020_freebayes.vcf.gz:
	bcftools view ${1KG_2020_FREEBAYES} | head -n 10000 | bcftools view -O z -o $@

1kg_2020_chr20_annotations.vcf.gz:
	bcftools view ${1KG_2020_ANNOTATIONS_URL} | head -n 100000 | bcftools view -O z -o $@

1kg_p3_all_chr1.vcf.gz:
	bcftools view ${1KG_P3_ALL_URL} | head -n 500000 | bcftools view -O z -o $@

1kg_p3_snp_chr1.vcf.gz:
	bcftools view ${1KG_P3_SNP_URL} | head -n 500000 | bcftools view -O z -o $@

1kg_p1_all_chr6.vcf.gz:
	bcftools view ${1KG_P1_ALL_URL} | head -n 10000 | bcftools view -O z -o $@

clean:
	rm -f *.vcf.gz *.tbi *.bcf *.csi
