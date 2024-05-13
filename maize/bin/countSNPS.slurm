#!/usr/bin/bash
#SBATCH --job-name=countSNP
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=40G
#SBATCH --time=45:00
#SBATCH --partition=jclarke,batch,guest,cse479,cse479_preempt
#SBATCH --output=/work/cse479/jdavis132/project/log/countSNP.out
#SBATCH --error=/work/cse479/jdavis132/project/log/countSNP.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jdavis132@huskers.unl.edu

cd /work/cse479/jdavis132/project/genotype
#wc -l combined.vcf >numSNPS.txt
#wc -l chr_1.vcf >numSNPS1.txt
#wc -l chr_2.vcf >numSNPS2.txt
#wc -l chr_3.vcf >numSNPS3.txt
#wc -l chr_4.vcf >numSNPS4.txt
#wc -l chr_5.vcf >numSNPS5.txt
#wc -l chr_6.vcf >numSNPS6.txt
#wc -l chr_7.vcf >numSNPS7.txt
#wc -l chr_8.vcf >numSNPS8.txt
#wc -l chr_9.vcf >numSNPS9.txt
#wc -l chr_10.vcf >numSNPS10.txt
#wc -l train.recode.vcf >numSNPStrain.txt
wc -l train_2.recode.vcf >numSNPS2.txt
wc -l train_3.recode.vcf >numSNPS3.txt
wc -l train_5.recode.vcf >numSNPS5.txt
wc -l train_6.recode.vcf >numSNPS6.txt
wc -l train_7.recode.vcf >numSNPS7.txt
wc -l train_8.recode.vcf >numSNPS8.txt
wc -l train_9.recode.vcf >numSNPS9.txt
wc -l train_10.recode.vcf >numSNPS10.txt
