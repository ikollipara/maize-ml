#!/usr/bin/bash
#SBATCH --job-name=transposeVCF
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=100G
#SBATCH --time=26:45:00
#SBATCH --partition=jclarke,batch,guest,cse479,cse479_preempt
#SBATCH --output=/work/cse479/jdavis132/project/log/transposeVCF.out
#SBATCH --error=/work/cse479/jdavis132/project/log/transposeVCF.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jdavis132@huskers.unl.edu

cd /work/cse479/jdavis132/project
#ml gzip
#gunzip genotype/chr_1.vcf.gz
#gunzip genotype/chr_2.vcf.gz
#gunzip genotype/chr_3.vcf.gz
#gunzip genotype/chr_4.vcf.gz
#gunzip genotype/chr_5.vcf.gz
#gunzip genotype/chr_6.vcf.gz
#gunzip genotype/chr_7.vcf.gz
#gunzip genotype/chr_8.vcf.gz
#gunzip genotype/chr_9.vcf.gz
#gunzip genotype/chr_10.vcf.gz

python src/transposeVCF.py
