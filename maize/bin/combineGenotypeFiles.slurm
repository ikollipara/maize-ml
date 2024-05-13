#!/usr/bin/bash
#SBATCH --job-name=combineGenotypeFiles
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=300G
#SBATCH --time=16:45:00
#SBATCH --partition=jclarke,batch,guest,cse479,cse479_preempt
#SBATCH --output=/work/cse479/jdavis132/project/log/combineGenotypeFiles.out
#SBATCH --error=/work/cse479/jdavis132/project/log/combineGenotypeFiles.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jdavis132@huskers.unl.edu

cd /work/cse479/jdavis132/project/genotype

#module load gunzip
ml tabix
module load bcftools
#ml tassel/5.2

#run_pipeline.pl -Xms512m -Xmx10G -fork1 -h hapmap/NAM_SNP_genos_raw_20090921_chr1.hmp -export -exportType VCF
#run_pipeline.pl -Xms512m -Xmx10G -fork1 -h hapmap/NAM_SNP_genos_raw_20090921_chr2.hmp -export -exportType VCF
#run_pipeline.pl -Xms512m -Xmx10G -fork1 -h hapmap/NAM_SNP_genos_raw_20090921_chr3.hmp -export -exportType VCF
#run_pipeline.pl -Xms512m -Xmx10G -fork1 -h hapmap/NAM_SNP_genos_raw_20090921_chr4.hmp -export -exportType VCF
#run_pipeline.pl -Xms512m -Xmx10G -fork1 -h hapmap/NAM_SNP_genos_raw_20090921_chr5.hmp -export -exportType VCF
#run_pipeline.pl -Xms512m -Xmx10G -fork1 -h hapmap/NAM_SNP_genos_raw_20090921_chr6.hmp -export -exportType VCF
#run_pipeline.pl -Xms512m -Xmx10G -fork1 -h hapmap/NAM_SNP_genos_raw_20090921_chr7.hmp -export -exportType VCF
#run_pipeline.pl -Xms512m -Xmx10G -fork1 -h hapmap/NAM_SNP_genos_raw_20090921_chr8.hmp -export -exportType VCF
#run_pipeline.pl -Xms512m -Xmx10G -fork1 -h hapmap/NAM_SNP_genos_raw_20090921_chr9.hmp -export -exportType VCF
#run_pipeline.pl -Xms512m -Xmx10G -fork1 -h hapmap/NAM_SNP_genos_raw_20090921_chr10.hmp -export -exportType VCF

#bgzip -c chr_1_imputed.vcf > chr_1_imputed.vcf.gz
#tabix -p vcf chr_1_imputed.vcf.gz

#bgzip -c chr_2_imputed.vcf > chr_2_imputed.vcf.gz
#tabix -p vcf chr_2_imputed.vcf.gz

#bgzip -c chr_3_imputed.vcf > chr_3_imputed.vcf.gz
#tabix -p vcf chr_3_imputed.vcf.gz

#bgzip -c chr_4_imputed.vcf > chr_4_imputed.vcf.gz
#tabix -p vcf chr_4_imputed.vcf.gz

#bgzip -c chr_5_imputed.vcf > chr_5_imputed.vcf.gz
#tabix -p vcf chr_5_imputed.vcf.gz

#bgzip -c chr_6_imputed.vcf > chr_6_imputed.vcf.gz
#tabix -p vcf chr_6_imputed.vcf.gz

#bgzip -c chr_7_imputed.vcf > chr_7_imputed.vcf.gz
#tabix -p vcf chr_7_imputed.vcf.gz

#bgzip -c chr_8_imputed.vcf > chr_8_imputed.vcf.gz
#tabix -p vcf chr_8_imputed.vcf.gz

#bgzip -c chr_9_imputed.vcf > chr_9_imputed.vcf.gz
#tabix -p vcf chr_9_imputed.vcf.gz

#bgzip -c chr_10_imputed.vcf > chr_10_imputed.vcf.gz
#tabix -p vcf chr_10_imputed.vcf.gz

#bcftools concat -o grzybowski.vcf.gz chr_1_imputed.vcf.gz chr_2_imputed.vcf.gz chr_3_imputed.vcf.gz chr_4_imputed.vcf.gz chr_5_imputed.vcf.gz chr_6_imputed.vcf.gz chr_7_imputed.vcf.gz chr_8_imputed.vcf.gz chr_9_imputed.vcf.gz chr_10_imputed.vcf.gz
#tabix -p vcf grzybowski.vcf.gz

#bgzip -c NAM_SNP_genos_raw_20090921_chr1.vcf > NAM_SNP_genos_raw_20090921_chr1.vcf.gz
#tabix -p vcf NAM_SNP_genos_raw_20090921_chr1.vcf.gz


#bgzip -c NAM_SNP_genos_raw_20090921_chr2.vcf > NAM_SNP_genos_raw_20090921_chr2.vcf.gz
#tabix -p vcf NAM_SNP_genos_raw_20090921_chr2.vcf.gz

#bgzip -c NAM_SNP_genos_raw_20090921_chr3.vcf > NAM_SNP_genos_raw_20090921_chr3.vcf.gz
#tabix -p vcf NAM_SNP_genos_raw_20090921_chr3.vcf.gz

#bgzip -c NAM_SNP_genos_raw_20090921_chr4.vcf > NAM_SNP_genos_raw_20090921_chr4.vcf.gz
#tabix -p vcf NAM_SNP_genos_raw_20090921_chr4.vcf.gz

#bgzip -c NAM_SNP_genos_raw_20090921_chr5.vcf > NAM_SNP_genos_raw_20090921_chr5.vcf.gz
#tabix -p vcf NAM_SNP_genos_raw_20090921_chr5.vcf.gz

#bgzip -c NAM_SNP_genos_raw_20090921_chr6.vcf > NAM_SNP_genos_raw_20090921_chr6.vcf.gz
#tabix -p vcf NAM_SNP_genos_raw_20090921_chr6.vcf.gz

#bgzip -c NAM_SNP_genos_raw_20090921_chr7.vcf > NAM_SNP_genos_raw_20090921_chr7.vcf.gz
#tabix -p vcf NAM_SNP_genos_raw_20090921_chr7.vcf.gz

#bgzip -c NAM_SNP_genos_raw_20090921_chr8.vcf > NAM_SNP_genos_raw_20090921_chr8.vcf.gz
#tabix -p vcf NAM_SNP_genos_raw_20090921_chr8.vcf.gz

#bgzip -c NAM_SNP_genos_raw_20090921_chr9.vcf > NAM_SNP_genos_raw_20090921_chr9.vcf.gz
#tabix -p vcf NAM_SNP_genos_raw_20090921_chr9.vcf.gz

#bgzip -c NAM_SNP_genos_raw_20090921_chr10.vcf > NAM_SNP_genos_raw_20090921_chr10.vcf.gz
#tabix -p vcf NAM_SNP_genos_raw_20090921_chr10.vcf.gz

#bcftools concat -o NAM.vcf.gz NAM_SNP_genos_raw_20090921_chr1.vcf.gz NAM_SNP_genos_raw_20090921_chr2.vcf.gz NAM_SNP_genos_raw_20090921_chr3.vcf.gz NAM_SNP_genos_raw_20090921_chr4.vcf.gz NAM_SNP_genos_raw_20090921_chr5.vcf.gz NAM_SNP_genos_raw_20090921_chr6.vcf.gz NAM_SNP_genos_raw_20090921_chr7.vcf.gz NAM_SNP_genos_raw_20090921_chr8.vcf.gz NAM_SNP_genos_raw_20090921_chr9.vcf.gz NAM_SNP_genos_raw_20090921_chr10.vcf.gz
#tabix -p vcf NAM.vcf.gz

bcftools merge --missing-to-ref --force-samples NAM.vcf.gz grzybowski.vcf.gz -o combined.vcf.gz
