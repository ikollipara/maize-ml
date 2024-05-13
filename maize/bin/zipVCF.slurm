#!/usr/bin/bash
#SBATCH --job-name=zipVCF
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=400G
#SBATCH --time=6:45:00
#SBATCH --partition=jclarke,batch,guest,cse479,cse479_preempt
#SBATCH --output=/work/cse479/jdavis132/project/log/zipVCF.out
#SBATCH --error=/work/cse479/jdavis132/project/log/zipVCF.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jdavis132@huskers.unl.edu

cd /work/cse479/jdavis132/project
ml bcftools
bgzip -c genotype/combined.vcf > genotype/combined.vcf.gz
