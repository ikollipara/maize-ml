#' 03_RRBLUP_analysis.R
#' Jensina Davis <jdavis132@huskers.unl.edu>
#' Ian Kollipara <ikollipara@huskers.unl.edu>

# Start the timer ------------------------------------------------------------
start <- proc.time()

# Load necessary libraries ---------------------------------------------------
library(rrBLUP)
library(tidyverse)

# Set file paths -------------------------------------------------------------
HAPMAP_TRAIN <- 'train.hmp.txt'
HAPMAP_TEST <- 'test.hmp.txt'
PHENO_TRAIN <- 'trainData.csv'
PHENO_TEST <- 'testData.csv'
WEATHER_DATA <- 'weatherData.csv'
DL_PRED <- 'predictions1_spear52.csv'

# train_df <- read.csv(PHENO_TRAIN) %>%
#   rename(GENOTYPE = genotypeVCF) %>%
#   group_by(GENOTYPE) %>%
#   summarise(earHeight = mean(earHeight, na.rm = TRUE)) %>%
#   arrange(GENOTYPE) %>%
#   filter(!is.na(earHeight))
#
# train_phenotypes <- unique(train_df$GENOTYPE)
#
# A_train <- read.table(HAPMAP_TRAIN, header = TRUE, sep = "\t")
# train_genotypes <- colnames(A_train)
#
# train_genotypes <- intersect(train_phenotypes, train_genotypes)
#
# A_train <- A_train %>%
#   select(c(alleles, all_of(train_genotypes))) %>%
#   rowwise() %>%
#   mutate(allele1 = str_split_i(alleles, fixed('/'), 1),
#          allele2 = str_split_i(alleles, fixed('/'), 2)) %>%
#   mutate(across(2:(length(train_genotypes) + 1), ~case_when(.==allele1 ~ 1, .default = 0))) %>%
#   sapply(as.numeric) %>%
#   as.matrix() %>%
#   t()
# A_train <- A_train[2:(dim(A_train)[1] - 2), ]
# A_train[is.na(A_train)] <- 0
#
# train_df <- train_df %>%
#   filter(GENOTYPE %in% train_genotypes) %>%
#   arrange(GENOTYPE) %>%
#   select(earHeight) %>%
#   as.matrix()
#
# trained_model <- mixed.solve(y=train_df, Z=A_train)
# marker_effects <- as.matrix(trained_model$u)
#
# test_df <- read.csv(PHENO_TEST) %>%
#   rename(GENOTYPE = genotypeVCF) %>%
#   group_by(GENOTYPE) %>%
#   summarise(earHeight = mean(earHeight, na.rm = TRUE)) %>%
#   arrange(GENOTYPE) %>%
#   filter(!is.na(earHeight))
#
# test_phenotypes <- unique(test_df$GENOTYPE)
#
# A_test <- read.table(HAPMAP_TEST, header = TRUE, sep = '\t')
# test_genotypes <- colnames(A_test)
#
# test_genotypes <- intersect(test_phenotypes, test_genotypes)
#
# A_test <- A_test %>%
#   select(c(alleles, all_of(test_genotypes))) %>%
#   rowwise() %>%
#   mutate(allele1 = str_split_i(alleles, fixed('/'), 1),
#          allele2 = str_split_i(alleles, fixed('/'), 2)) %>%
#   mutate(across(2:(length(test_genotypes) + 1), ~case_when(.==allele1 ~ 1, .default = 0))) %>%
#   sapply(as.numeric) %>%
#   as.matrix() %>%
#   t()
#
# A_test <- A_test[2:(dim(A_test)[1] - 2), ]
# A_test[is.na(A_test)] <- 0
#
# test_df <- test_df %>%
#   filter(GENOTYPE %in% test_genotypes) %>%
#   arrange(GENOTYPE)
#
# test_pred <- as.matrix(A_test) %*% marker_effects
# test_df <- bind_cols(test_df, test_pred)
# colnames(test_df) <- c('genotype', 'obs', 'pred')
# write.csv(test_df, 'rrblupTestPerformance.csv', quote = FALSE, row.names = FALSE)
#
# plot <- ggplot(test_df, aes(obs, pred)) +
#   geom_point()
# plot

# CV

weather <- read.csv(WEATHER_DATA)[, 2:11] %>%
  mutate(across(where(is.numeric), ~case_when(.==-Inf|.==Inf ~ NA, .default = .))) %>%
  group_by(sourceEnvironment, year) %>%
  summarise(dayl = sum(dayl..s., na.rm = TRUE),
            prcp = sum(prcp..mm.day., na.rm = TRUE),
            srad = sum(srad..W.m.2., na.rm = TRUE),
            swe = sum(swe..kg.m.2., na.rm = TRUE),
            tmax = sum(tmax..deg.c., na.rm = TRUE),
            tmin = sum(tmin..deg.c., na.rm = TRUE),
            vp = sum(vp..Pa., na.rm = TRUE)) %>%
  ungroup() %>%
  group_by(sourceEnvironment) %>%
  summarise(dayl = mean(dayl, na.rm = TRUE),
            prcp = mean(prcp, na.rm = TRUE),
            srad = mean(srad, na.rm = TRUE),
            swe = mean(swe, na.rm = TRUE),
            tmax = mean(tmax, na.rm = TRUE),
            tmin = mean(tmin, na.rm = TRUE),
            vp = mean(vp, na.rm = TRUE))

weather_pca <- princomp(as.matrix(weather[, 2:8]), scores = TRUE)
# Scree plot
screeData <- weather_pca$sdev
totalVar <- sum(screeData)
screeData <- tibble(component = 1:7, variance = screeData, percentVariance = (variance/totalVar)*100)
screePlot <- ggplot(screeData, aes(component, percentVariance)) +
  geom_line() +
  labs(x = 'Principle Component', y = 'Variance Explained (%)') +
  theme_minimal() +
  theme(axis.text.x = element_text(color = 'black', size = 10),
        axis.text.y = element_text(color = 'black', size = 10),
        legend.text = element_text(color = 'black', size = 10),
        text = element_text(color = 'black', size = 10),
        panel.grid = element_blank())
screePlot

pca_scores <- bind_cols(weather[, 1], weather_pca$scores[, 1:3])

pheno_df1 <- read.csv(PHENO_TRAIN) %>%
    rename(GENOTYPE = genotypeVCF) %>%
    group_by(GENOTYPE, sourceEnvironment) %>%
    summarise(earHeight = mean(earHeight, na.rm = TRUE)) %>%
    arrange(sourceEnvironment, GENOTYPE) %>%
    filter(!is.na(earHeight))

pheno_df2 <- read.csv(PHENO_TEST) %>%
    rename(GENOTYPE = genotypeVCF) %>%
    group_by(GENOTYPE, sourceEnvironment) %>%
    summarise(earHeight = mean(earHeight, na.rm = TRUE)) %>%
    arrange(sourceEnvironment, GENOTYPE) %>%
    filter(!is.na(earHeight))

pheno_df <- bind_rows(pheno_df1, pheno_df2)
pheno_df <- left_join(pheno_df, pca_scores, join_by(sourceEnvironment), keep = FALSE, relationship = 'many-to-one')
pheno_genotypes <- unique(pheno_df$GENOTYPE)

A_1 <- read.table(HAPMAP_TRAIN, header = TRUE, sep = "\t")
A_2 <- read.table(HAPMAP_TEST, header = TRUE, sep = '\t')
A_2 <- A_2[, 12:length(colnames(A_2))]

A <- bind_cols(A_1, A_2)
A_genotypes <- colnames(A)

genotypes <- intersect(pheno_genotypes, A_genotypes)

pheno_df <- pheno_df %>%
  filter(GENOTYPE %in% genotypes) %>%
  arrange(GENOTYPE)

A <- A %>%
    select(c(alleles, all_of(genotypes))) %>%
    rowwise() %>%
    mutate(allele1 = str_split_i(alleles, fixed('/'), 1),
           allele2 = str_split_i(alleles, fixed('/'), 2)) %>%
    mutate(across(2:(length(genotypes) + 1), ~case_when(.==allele1 ~ 1, .default = 0))) %>%
    sapply(as.numeric) %>%
    as.matrix() %>%
    t()
A <- A[2:(dim(A)[1] - 2), ]
A[is.na(A)] <- 0

# # CV by genotype
# FOLD_SIZE <- round(0.2*length(genotypes))
#
# fold1 <- sample(genotypes, size = FOLD_SIZE)
# remaining <- setdiff(genotypes, fold1)
# fold2 <- sample(remaining, size = FOLD_SIZE)
# remaining <- setdiff(remaining, fold2)
# fold3 <- sample(remaining, size = FOLD_SIZE)
# remaining <- setdiff(remaining, fold3)
# fold4 <- sample(remaining, size = FOLD_SIZE)
# fold5 <- setdiff(remaining, fold4)

# test_pred <- tibble(GENOTYPE = NULL, sourceEnvironment = NULL, obs = NULL, pred = NULL)
# for(fold in folds)
# {
#   train_genotypes <- setdiff(genotypes, fold)
#
#   train_df <- pheno_df %>%
#     filter(GENOTYPE %in% train_genotypes) %>%
#     arrange(GENOTYPE)
#
#   Z_train <- matrix(0, nrow = length(train_df$GENOTYPE), ncol = 10000)
#   for(i in 1:length(train_df$GENOTYPE))
#   {
#     genotype <- train_df$GENOTYPE[i]
#     Z_train[i, ] <- A[genotype, ]
#   }
#
#   train_df <- train_df %>%
#     ungroup() %>%
#     select(earHeight, Comp.1, Comp.2, Comp.3) %>%
#     as.matrix()
#
#   trained_model <- mixed.solve(y=train_df[, 1], X = train_df[, 2:4], Z=Z_train)
#   print('trained')
#   weather_effects <- as.matrix(trained_model$beta)
#   marker_effects <- as.matrix(trained_model$u)
#
#   test_df <- pheno_df %>%
#     filter(GENOTYPE %in% fold) %>%
#     arrange(GENOTYPE) %>%
#     select(GENOTYPE, sourceEnvironment, earHeight, Comp.1, Comp.2, Comp.3)
#
#   Z_test <- matrix(0, nrow = length(test_df$GENOTYPE), ncol = 10000)
#   for(i in 1:length(test_df$GENOTYPE))
#   {
#     Z_test[i, ] <- A[test_df$GENOTYPE[i], ]
#   }
#
#   pred <- as.matrix(Z_test) %*% marker_effects + as.matrix(test_df[, 4:6]) %*% weather_effects
#
#   test_df <- test_df %>%
#     select(GENOTYPE, sourceEnvironment, earHeight) %>%
#     bind_cols(pred)
#   colnames(test_df) <- c('GENOTYPE', 'sourceEnvironment', 'obs', 'pred')
#
#   test_pred <- bind_rows(test_pred, test_df)
# }
start <- proc.time()
# CV by environment
environments <- unique(pheno_df$sourceEnvironment)
FOLD_SIZE <- round(0.2*length(environments))
fold1 <- sample(environments, size = FOLD_SIZE)
remaining <- setdiff(environments, fold1)
fold2 <- sample(remaining, size = FOLD_SIZE)
remaining <- setdiff(environments, fold2)
fold3 <- sample(remaining, size = FOLD_SIZE)
remaining <- setdiff(remaining, fold3)
fold4 <- sample(remaining, size = FOLD_SIZE)
fold5 <- setdiff(remaining, fold4)

folds <- list(fold1, fold2, fold3, fold4, fold5)

test_pred <- tibble(GENOTYPE = NULL, sourceEnvironment = NULL, obs = NULL, pred = NULL)
for(fold in folds)
{
  train_environments <- setdiff(environments, fold)

  train_df <- pheno_df %>%
    filter(sourceEnvironment %in% train_environments) %>%
    arrange(sourceEnvironment)

  Z_train <- matrix(0, nrow = length(train_df$GENOTYPE), ncol = 10000)
  for(i in 1:length(train_df$GENOTYPE))
  {
    genotype <- train_df$GENOTYPE[i]
    Z_train[i, ] <- A[genotype, ]
  }

  train_df <- train_df %>%
    ungroup() %>%
    select(earHeight, Comp.1, Comp.2, Comp.3) %>%
    as.matrix()

  trained_model <- mixed.solve(y=train_df[, 1], X = train_df[, 2:4], Z=Z_train)
  print('trained')
  weather_effects <- as.matrix(trained_model$beta)
  marker_effects <- as.matrix(trained_model$u)

  test_df <- pheno_df %>%
    filter(sourceEnvironment %in% fold) %>%
    arrange(sourceEnvironment) %>%
    select(GENOTYPE, sourceEnvironment, earHeight, Comp.1, Comp.2, Comp.3)

  Z_test <- matrix(0, nrow = length(test_df$GENOTYPE), ncol = 10000)
  for(i in 1:length(test_df$GENOTYPE))
  {
    Z_test[i, ] <- A[test_df$GENOTYPE[i], ]
  }

  pred <- as.matrix(Z_test) %*% marker_effects + as.matrix(test_df[, 4:6]) %*% weather_effects

  test_df <- test_df %>%
    select(GENOTYPE, sourceEnvironment, earHeight) %>%
    bind_cols(pred)
  colnames(test_df) <- c('GENOTYPE', 'sourceEnvironment', 'obs', 'pred')

  test_pred <- bind_rows(test_pred, test_df)
}

dl <- read.csv(DL_PRED)

legendData <- tibble(x = rep(1, 2), y = rep(1, 2), model = c('GBLUP', 'DL'))
legend <- ggplot(legendData, aes(x, y, color = model)) +
  geom_point() +
  scale_color_manual(values = c('red', 'blue')) +
  labs(color = 'Model') +
  theme_minimal() +
  theme(axis.text.x = element_text(color = 'black', size = 14),
        axis.text.y = element_text(color = 'black', size = 14),
        legend.text = element_text(color = 'black', size = 14),
        text = element_text(color = 'black', size = 14),
        panel.grid = element_blank())
legend <- get_legend(legend)

gblup_r <- cor(test_pred$obs, test_pred$pred, use = 'complete.obs', method = 'spearman')
dl_r <- cor(dl$Actual, dl$Predicted, use = 'complete.obs', method = 'spearman')
plot_cv <- ggplot() +
  geom_point(aes(obs, pred), test_pred, color = 'blue', show.legend = TRUE) +
  geom_smooth(aes(obs, pred), test_pred, method = 'lm', color = 'blue', show.legend = TRUE) +
  geom_point(aes(Actual, Predicted), dl, color = 'red', show.legend = TRUE) +
  geom_smooth(aes(Actual, Predicted), dl, method = 'lm', color = 'red', show.legend = TRUE) +
  # geom_hline(yintercept = mean(dl$Actual)) +
  # geom_hline(yintercept = (mean(dl$Actual) + sd(dl$Actual))) +
  # geom_hline(yintercept = (mean(dl$Actual) - sd(dl$Actual))) +
  scale_x_continuous(limits = c(0, 200)) +
  scale_y_continuous(limits = c(0, 200)) +
  labs(x = 'Observed Ear Height (cm)', y = 'Predicted Ear Height (cm)',
       subtitle = paste0(
       'GBLUP Spearman Rank Correlation Coefficient: ', round(gblup_r, 4), '
DL Spearman Rank Correlation Coefficient: ', round(dl_r, 4))) +
  theme_minimal() +
  theme(axis.text.x = element_text(color = 'black', size = 14),
        axis.text.y = element_text(color = 'black', size = 14),
        legend.text = element_text(color = 'black', size = 14),
        text = element_text(color = 'black', size = 14),
        panel.grid = element_blank())
plot_cv

dl_hist <- ggplot(dl) +
  geom_histogram(aes(Predicted), color = 'blue') +
  geom_histogram(aes(Actual), color = 'red')
dl_hist

end <- proc.time()
print(paste('Time used: ', (end - start)))


# write.csv(test_pred, 'gblup_performance_20240510A.csv')

gblup_lower <- read.csv('gblup_performance_20240510A.csv')
gblup_r <- cor(gblup_lower$obs, gblup_lower$pred, use = 'complete.obs', method = 'spearman')
plot_cv <- ggplot() +
  geom_point(aes(obs, pred), gblup_lower, color = 'blue', show.legend = TRUE) +
  geom_smooth(aes(obs, pred), gblup_lower, method = 'lm', color = 'blue', show.legend = TRUE) +
  geom_point(aes(Actual, Predicted), dl, color = 'red', show.legend = TRUE) +
  geom_smooth(aes(Actual, Predicted), dl, method = 'lm', color = 'red', show.legend = TRUE) +
  # geom_hline(yintercept = mean(dl$Actual)) +
  # geom_hline(yintercept = (mean(dl$Actual) + sd(dl$Actual))) +
  # geom_hline(yintercept = (mean(dl$Actual) - sd(dl$Actual))) +
  scale_x_continuous(limits = c(0, 200)) +
  scale_y_continuous(limits = c(0, 200)) +
  labs(x = 'Observed Ear Height (cm)', y = 'Predicted Ear Height (cm)',
       subtitle = paste0(
         'GBLUP Spearman Rank Correlation Coefficient: ', round(gblup_r, 4), '
DL Spearman Rank Correlation Coefficient: ', round(dl_r, 4))) +
  theme_minimal() +
  theme(axis.text.x = element_text(color = 'black', size = 14),
        axis.text.y = element_text(color = 'black', size = 14),
        legend.text = element_text(color = 'black', size = 14),
        text = element_text(color = 'black', size = 14),
        panel.grid = element_blank())
plot_cv
