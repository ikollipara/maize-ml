#!/usr/bin/bash
#SBATCH --job-name=splitVCFGBLUP
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=50G
#SBATCH --time=26:45:00
#SBATCH --partition=jclarke,batch,guest,cse479,cse479_preempt
#SBATCH --output=/work/cse479/jdavis132/project/log/splitVCFGBLUP.out
#SBATCH --error=/work/cse479/jdavis132/project/log/splitVCFGBLUP.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jdavis132@huskers.unl.edu

cd /work/cse479/jdavis132/project/genotype
ml vcftools

vcftools --vcf chr2_downsample.vcf --out train_2 --keep train_genos.txt --recode
vcftools --vcf chr3_downsample.vcf --out train_3 --keep train_genos.txt --recode
vcftools --vcf chr5_downsample.vcf --out train_5 --keep train_genos.txt --recode
vcftools --vcf chr6_downsample.vcf --out train_6 --keep train_genos.txt --recode
vcftools --vcf chr7_downsample.vcf --out train_7 --keep train_genos.txt --recode
vcftools --vcf chr8_downsample.vcf --out train_8 --keep train_genos.txt --recode
vcftools --vcf chr9_downsample.vcf --out train_9 --keep train_genos.txt --recode
vcftools --vcf chr10_downsample.vcf --out train_10 --keep train_genos.txt --recode

vcftools --vcf chr2_downsample.vcf --out test_2 --keep test_genos.txt --recode
vcftools --vcf chr3_downsample.vcf --out test_3 --keep test_genos.txt --recode
vcftools --vcf chr5_downsample.vcf --out test_5 --keep test_genos.txt --recode
vcftools --vcf chr6_downsample.vcf --out test_6 --keep test_genos.txt --recode
vcftools --vcf chr7_downsample.vcf --out test_7 --keep test_genos.txt --recode
vcftools --vcf chr8_downsample.vcf --out test_8 --keep test_genos.txt --recode
vcftools --vcf chr9_downsample.vcf --out test_9 --keep test_genos.txt --recode
vcftools --vcf chr10_downsample.vcf --out test_10 --keep test_genos.txt --recode
