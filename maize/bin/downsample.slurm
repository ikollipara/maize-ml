#!/usr/bin/bash
#SBATCH --job-name=downsample
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=500G
#SBATCH --time=26:45:00
#SBATCH --partition=jclarke,batch,guest,cse479,cse479_preempt
#SBATCH --output=/work/cse479/jdavis132/project/log/downsample.out
#SBATCH --error=/work/cse479/jdavis132/project/log/downsample.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jdavis132@huskers.unl.edu

cd /work/cse479/jdavis132/project/genotype

ml tassel/5.2

run_pipeline.pl -Xms400G -Xmx400G -fork1 -vcf chr_1_filt.recode.vcf -subsetSites 0.1 -export chr1_downsample.vcf -exportType VCF
run_pipeline.pl -Xms400G -Xmx400G -fork1 -vcf chr_2_filt.recode.vcf -subsetSites 0.1 -export chr2_downsample.vcf -exportType VCF
run_pipeline.pl -Xms400G -Xmx400G -fork1 -vcf chr_3_filt.recode.vcf -subsetSites 0.1 -export chr3_downsample.vcf -exportType VCF
run_pipeline.pl -Xms400G -Xmx400G -fork1 -vcf chr_4_filt.recode.vcf -subsetSites 0.1 -export chr4_downsample.vcf -exportType VCF
run_pipeline.pl -Xms400G -Xmx400G -fork1 -vcf chr_5_filt.recode.vcf -subsetSites 0.1 -export chr5_downsample.vcf -exportType VCF
run_pipeline.pl -Xms400G -Xmx400G -fork1 -vcf chr_6_filt.recode.vcf -subsetSites 0.1 -export chr6_downsample.vcf -exportType VCF
run_pipeline.pl -Xms400G -Xmx400G -fork1 -vcf chr_7_filt.recode.vcf -subsetSites 0.1 -export chr7_downsample.vcf -exportType VCF
run_pipeline.pl -Xms400G -Xmx400G -fork1 -vcf chr_8_filt.recode.vcf -subsetSites 0.1 -export chr8_downsample.vcf -exportType VCF
run_pipeline.pl -Xms400G -Xmx400G -fork1 -vcf chr_9_filt.recode.vcf -subsetSites 0.1 -export chr9_downsample.vcf -exportType VCF
run_pipeline.pl -Xms400G -Xmx400G -fork1 -vcf chr_10_filt.recode.vcf -subsetSites 0.1 -export chr10_downsample.vcf -exportType VCF

