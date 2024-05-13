#!/usr/bin/bash
#SBATCH --job-name=unzipVCF
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=10G
#SBATCH --time=45:00
#SBATCH --partition=jclarke,batch,guest
#SBATCH --output=/work/cse479/jdavis132/project/log/unzipVCF.out
#SBATCH --error=/work/cse479/jdavis132/project/log/unzipVCF.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jdavis132@huskers.unl.edu

cd /work/cse479/jdavis132/project/genotype

module load gzip

gunzip chr_1_imputed.vcf.gz
gunzip chr_2_imputed.vcf.gz
gunzip chr_3_imputed.vcf.gz
gunzip chr_4_imputed.vcf.gz
gunzip chr_5_imputed.vcf.gz
gunzip chr_6_imputed.vcf.gz
gunzip chr_7_imputed.vcf.gz
gunzip chr_8_imputed.vcf.gz
gunzip chr_9_imputed.vcf.gz
gunzip chr_10_imputed.vcf.gz
