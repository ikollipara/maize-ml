#' 02_split_genotypes.R
#' Jensina Davis <jdavis132@huskers.unl.edu>
#' Ian Kollipara <ikollipara2@huskers.unl.edu>

# Load necessary libraries ---------------------------------------------------
library(tidyverse)

# Load data -------------------------------------------------------------------
vcf_genos <- read.table("../combinedGenotypes.txt", row.names=NULL, quote="", comment.char="")[, 10:6250] %>%
  as_vector()
genotype <- str_to_upper(vcf_genos)

vcf <- tibble(genotype = genotype, genotypeVCF = vcf_genos) %>%
       group_by(genotype) %>%
       mutate(n = n()) %>%
       filter(n==1) %>%
       select(!n)

earheight <- read.csv('../earHeight.csv')

phenotype <- full_join(earheight, vcf, join_by(genotype), keep = FALSE, suffix = c('', ''), relationship =  'many-to-one') %>%
             filter(!is.na(genotypeVCF)) %>%
             filter(!str_detect(genotypeVCF, 'Z0'))

genotypes <- unique(phenotype$genotype)

train_genos <- sample(genotypes, size = round(0.8*length(genotypes)))
test_genos <- setdiff(genotypes, train_genos)

test_data <- phenotype %>% filter(genotype %in% test_genos)
train_data <- phenotype %>% filter(genotype %in% train_genos)

all_genotypes <- c(unique(test_data$genotypeVCF), unique(train_data$genotypeVCF))

# Write data ------------------------------------------------------------------

write.csv(test_data, 'testData.csv', row.names = FALSE)
write.csv(train_data, 'trainData.csv', row.names = FALSE)

write.table(unique(test_data$genotypeVCF), 'test_genos.txt', row.names = FALSE, sep = '\t', quote = FALSE)
write.table(unique(train_data$genotypeVCF), 'train_genos.txt', row.names = FALSE, sep = '\t', quote = FALSE)
write.table(all_genotypes, 'allGenotypesInPhenotype.txt', row.names = FALSE, sep = '\t', quote = FALSE)
