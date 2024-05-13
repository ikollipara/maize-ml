#!/usr/bin/bash
#SBATCH --job-name=splitVCF
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=400G
#SBATCH --time=6:45:00
#SBATCH --partition=jclarke,batch,guest,cse479,cse479_preempt
#SBATCH --output=/work/cse479/jdavis132/project/log/splitVCF.out
#SBATCH --error=/work/cse479/jdavis132/project/log/splitVCF.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jdavis132@huskers.unl.edu

cd /work/cse479/jdavis132/project/genotype
ml bcftools
#ml tabix
#ml gatk4

#tabix -p vcf combined.vcf.gz

#bcftools filter -r 1 -o chr_1.vcf.gz combined.vcf.gz
#bcftools filter -r 2 -o chr_2.vcf.gz combined.vcf.gz
#bcftools filter -r 3 -o chr_3.vcf.gz combined.vcf.gz
#bcftools filter -r 4 -o chr_4.vcf.gz combined.vcf.gz
#bcftools filter -r 5 -o chr_5.vcf.gz combined.vcf.gz
#bcftools filter -r 6 -o chr_6.vcf.gz combined.vcf.gz
#bcftools filter -r 7 -o chr_7.vcf.gz combined.vcf.gz
#bcftools filter -r 8 -o chr_8.vcf.gz combined.vcf.gz
#bcftools filter -r 9 -o chr_9.vcf.gz combined.vcf.gz
#bcftools filter -r 10 -o chr_10.vcf.gz combined.vcf.gz

#gatk SelectVariants -V combined.vcf.gz -L 1 -O chr_1.vcf.gz
#gatk SelectVariants -V combined.vcf.gz -L 2 -O chr_2.vcf.gz
#gatk SelectVariants -V combined.vcf.gz -L 3 -O chr_3.vcf.gz
#gatk SelectVariants -V combined.vcf.gz -L 4 -O chr_4.vcf.gz
#gatk SelectVariants -V combined.vcf.gz -L 5 -O chr_5.vcf.gz
#gatk SelectVariants -V combined.vcf.gz -L 6 -O chr_6.vcf.gz
#gatk SelectVariants -V combined.vcf.gz -L 7 -O chr_7.vcf.gz
#gatk SelectVariants -V combined.vcf.gz -L 8 -O chr_8.vcf.gz
#gatk SelectVariants -V combined.vcf.gz -L 9 -O chr_9.vcf.gz
#gatk SelectVariants -V combined.vcf.gz -L 10 -O chr_10.vcf.gz

bcftools filter -O v -o chr_1.vcf -r 'chr1' combined.vcf.gz
bcftools filter -O v -o chr_2.vcf -r 'chr2' combined.vcf.gz
bcftools filter -O v -o chr_3.vcf -r 'chr3' combined.vcf.gz
bcftools filter -O v -o chr_4.vcf -r 'chr4' combined.vcf.gz
bcftools filter -O v -o chr_5.vcf -r 'chr5' combined.vcf.gz
bcftools filter -O v -o chr_6.vcf -r 'chr6' combined.vcf.gz
bcftools filter -O v -o chr_7.vcf -r 'chr7' combined.vcf.gz
bcftools filter -O v -o chr_8.vcf -r 'chr8' combined.vcf.gz
bcftools filter -O v -o chr_9.vcf -r 'chr9' combined.vcf.gz
bcftools filter -O v -o chr_10.vcf -r 'chr10' combined.vcf.gz
