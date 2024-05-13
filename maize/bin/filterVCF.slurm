#!/usr/bin/bash
#SBATCH --job-name=splitVCF
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=400G
#SBATCH --time=16:45:00
#SBATCH --partition=jclarke,batch,guest,cse479,cse479_preempt
#SBATCH --output=/work/cse479/jdavis132/project/log/splitVCF.out
#SBATCH --error=/work/cse479/jdavis132/project/log/splitVCF.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jdavis132@huskers.unl.edu

cd /work/cse479/jdavis132/project/genotype
ml vcftools

#vcftools --vcf chr_1.vcf --out chr_1_filt --keep allGenotypesInPhenotype.txt --recode
#vcftools --vcf chr_2.vcf --out chr_2_filt --keep allGenotypesInPhenotype.txt --recode
#vcftools --vcf chr_3.vcf --out chr_3_filt --keep allGenotypesInPhenotype.txt --recode
#vcftools --vcf chr_4.vcf --out chr_4_filt --keep allGenotypesInPhenotype.txt --recode
#vcftools --vcf chr_5.vcf --out chr_5_filt --keep allGenotypesInPhenotype.txt --recode
#vcftools --vcf chr_6.vcf --out chr_6_filt --keep allGenotypesInPhenotype.txt --recode
#vcftools --vcf chr_7.vcf --out chr_7_filt --keep allGenotypesInPhenotype.txt --recode
vcftools --vcf chr_8.vcf --out chr_8_filt --keep allGenotypesInPhenotype.txt --recode
vcftools --vcf chr_9.vcf --out chr_9_filt --keep allGenotypesInPhenotype.txt --recode
vcftools --vcf chr_10.vcf --out chr_10_filt --keep allGenotypesInPhenotype.txt --recode
