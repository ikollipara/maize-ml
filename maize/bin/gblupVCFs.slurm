#!/usr/bin/bash
#SBATCH --job-name=VCFGBLUP
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=200G
#SBATCH --time=26:45:00
#SBATCH --partition=jclarke,batch,guest,cse479,cse479_preempt
#SBATCH --output=/work/cse479/jdavis132/project/log/VCFGBLUP.out
#SBATCH --error=/work/cse479/jdavis132/project/log/VCFGBLUP.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jdavis132@huskers.unl.edu

cd /work/cse479/jdavis132/project/genotype
#ml bcftools
#ml tabix
#ml bgzip
#ml gzip

#bgzip -c train_2.recode.vcf > train_2.vcf.gz
#tabix train_2.vcf.gz

#bgzip -c train_3.recode.vcf > train_3.vcf.gz
#tabix train_3.vcf.gz

#bgzip  train_5.recode.vcf > train_5.vcf.gz
#tabix  train_5.vcf.gz

#bgzip -c train_6.recode.vcf > train_6.vcf.gz
#tabix train_6.vcf.gz

#bgzip -c train_7.recode.vcf > train_7.vcf.gz
#tabix train_7.vcf.gz

#bgzip -c train_8.recode.vcf > train_8.vcf.gz
#tabix train_8.vcf.gz

#bgzip -c train_9.recode.vcf > train_9.vcf.gz
#tabix train_9.vcf.gz

#bgzip -c train_10.recode.vcf > train_10.vcf.gz
#tabix train_10.vcf.gz

#bgzip -c test_2.recode.vcf > test_2.vcf.gz
#tabix test_2.vcf.gz

#bgzip -c test_3.recode.vcf > test_3.vcf.gz
#tabix test_3.vcf.gz

#bgzip test_5.recode.vcf > test_5.vcf.gz
#tabix test_5.vcf.gz

#bgzip -c test_6.recode.vcf > test_6.vcf.gz
#tabix test_6.vcf.gz

#bgzip -c test_7.recode.vcf > test_7.vcf.gz
#tabix test_7.vcf.gz

#bgzip -c test_8.recode.vcf > test_8.vcf.gz
#tabix test_8.vcf.gz

#bgzip -c test_9.recode.vcf > test_9.vcf.gz
#tabix test_9.vcf.gz

#bgzip -c test_10.recode.vcf > test_10.vcf.gz
#tabix test_10.vcf.gz

#bcftools concat -o train.vcf.gz train_2.vcf.gz train_3.vcf.gz train_5.vcf.gz train_6.vcf.gz train_7.vcf.gz train_8.vcf.gz train_9.vcf.gz train_10.vcf.gz
#bcftools concat -o test.vcf.gz test_2.vcf.gz test_3.vcf.gz test_5.vcf.gz test_6.vcf.gz test_7.vcf.gz test_8.vcf.gz test_9.vcf.gz test_10.vcf.gz

#gunzip train.vcf.gz
#gunzip test.vcf.gz

head -n 10011 train_2.recode.vcf >train.vcf
head -n 10011 test_2.recode.vcf >test.vcf 
ml tassel/5.2
run_pipeline.pl -Xms150G -Xmx150G -fork1 -vcf train.vcf -export -export -exportType Hapmap
run_pipeline.pl -Xms150G -Xmx150G -fork1 -vcf test.vcf -export -export -exportType Hapmap
