#' 01_build_dataset.R
#' Jensina Davis <jdavis132@huskers.unl.edu>
#' Ian Kollipara <ikollipara2@huskers.unl.edu>

# Loading Libraries --------------------------------------------
library(tidyverse)
library(daymetr)
library(googleway)

# Load Data -----------------------------------------------------
panzea_traits <- as.character(read.table('rawData/traitMatrix_maize282NAM_v15-130212.txt', nrows = 1)[1, 2:286])
panzea_envs <- as.character(read.table('rawData/traitMatrix_maize282NAM_v15-130212.txt', skip = 1, nrows = 1)[1, 3:287])
panzea_data <- read.table('rawData/traitMatrix_maize282NAM_v15-130212.txt', skip = 2)

# Wrangle Data --------------------------------------------------
panzea_data <- panzea_data %>%%
               pivot_longer(!c(genotype), names_to = 'environment', values_to = 'earHeight') %>%
               rowwise() %>%
               mutate(
                genotype = str_to_upper(genotype),
                environment = str_to_upper(environment)
               ) %>%
               mutate(
                environment = case_when(environment=='65' ~ '065', .default = environment)
               ) %>%
               filter(!(genotype %in% c('B73_MO17', 'MO17_B73')))

panzea_environment_codes <- read.csv('rawData/panzeaEnvironmentCodes.csv') %>%
                           rowwise() %>%
                           mutate(
                            environment = str_to_upper(envCode),
                            lat = case_when(Latitude=='NULL' ~ NA, .default = Latitude) %>%
                              as.numeric(),
                            lon = case_when(Longitude=='NULL' ~ NA, .default = Longitude) %>%
                              as.numeric()*-1
                           ) %>%
                           rename(
                            city = City,
                            state = State
                           ) %>%
                           select(environment, lat, lon, city, state)

panzea_data <- full_join(
    panzea_data,
    panzea_environment_codes,
    join_by(environment),
    keep = FALSE,
    suffix = c('', '')
    ) %>%
    mutate(source = 'Panzea')

# Generate the Datasets -----------------------------------------

g2f2014 <- read.csv('rawData/g2f_2014_inbred_data_clean.csv') %>%
  rename(city = City,
         genotype = Genotype,
         environment = Experiment) %>%
  rowwise() %>%
  mutate(state = str_split_i(Location, '_', 3) %>%
           str_sub(1, 2))
g2f2014Meta <- read.csv('rawData/g2f_2014_field_characteristics.csv') %>%
  filter(Type=='inbred') %>%
  rowwise() %>%
  mutate(Experiment  = str_replace(Experiment, 'G2F', 'GXE_inb_')) %>%
  mutate(Experiment = case_when(Experiment=='NC1' ~ 'GXE_inb_NC1',
                                Experiment=='GxE_inb_PA1' ~ 'GXE_inb_PA1',
                                Experiment=='GXE_inb_WI1' ~ 'GXE_inb_WI1_MAD',
                                Experiment=='GXE_inb_WI2' ~ 'GXE_inb_WI2_ARL',
                                .default = Experiment)) %>%
  rename(environment = Experiment,
         city = City,
         lon = long) %>%
  select(environment, lat, lon, city)

g2f2014 <- full_join(g2f2014, g2f2014Meta, join_by(environment), keep = FALSE, suffix = c('', ''), relationship = 'many-to-one') %>%
  select(environment, city, genotype, earHeight, state, lat, lon) %>%
  mutate(source = 'G2F_2014')

g2f2015 <- read.csv('rawData/g2f_2015_inbred_data_clean.csv') %>%
  rename(environment = Book.Name,
         state = State,
         genotype = Pedigree,
         earHeight = Ear.height..cm.) %>%
  select(environment, state, genotype, earHeight)

g2f2015Meta <- read.csv('rawData/g2f_2015_field_metadata.csv') %>%
  filter(Type=='Inbred') %>%
  rename(environment = Experiment,
         city = City,
         lat = corner1.lat,
         lon = corner1.lon) %>%
  select(environment, city, lat, lon)

g2f2015 <- full_join(g2f2015, g2f2015Meta, join_by(environment), keep = FALSE, suffix = c('', ''), relationship = 'many-to-one')  %>%
  mutate(source = 'G2F_2015')

g2f2020 <- read.csv('rawData/g2f_2020_phenotypic_clean_data.csv')%>%
  filter(Experiment=='HIP_Inbred') %>%
  rename(environment = Field.Location,
         state = State,
         earHeight = Ear.Height..cm.,
         genotype = Pedigree) %>%
  select(environment, state, earHeight, genotype)

g2f2020Meta <- read.csv('rawData/g2f_2020_field_metadata.csv') %>%
  rename(environment = Experiment_Code,
         city = City,
         lat = Latitude_of_Field_Corner_.1..lower.left.,
         lon = Longitude_of_Field_Corner_.1..lower.left.) %>%
  select(environment, city, lat, lon)

g2f2020 <- full_join(g2f2020, g2f2020Meta, join_by(environment), keep = FALSE, suffix = c('', ''), relationship = 'many-to-one') %>%
  mutate(source = 'G2F_2020')

g2f2021 <- read.csv('rawData/g2f_2021_phenotypic_clean_data.csv') %>%
  filter(Experiment=='HIP_Inbred') %>%
  rename(environment = Field.Location,
         genotype = Pedigree,
         earHeight = Ear.Height..cm.) %>%
  rowwise() %>%
  mutate(state = str_remove(State, '\n')) %>%
  select(environment, genotype, earHeight, state)

g2f2021Meta <- read.csv('rawData/g2f_2021_field_metadata.csv') %>%
  rename(environment = Experiment_Code,
         city = City,
         lat = Latitude_of_Field_Corner_.1..lower.left.,
         lon = Longitude_of_Field_Corner_.1..lower.left.) %>%
  select(environment, city, lat, lon)

g2f2021 <- full_join(g2f2021, g2f2021Meta, join_by(environment), keep = FALSE, suffix = c('', ''), relationship = 'many-to-one') %>%
  mutate(source = 'G2F_2021')

g2f2022 <- read.csv('rawData/g2f_2022_phenotypic_clean_data.csv') %>%
  filter(Experiment=='HIP_Inbred') %>%
  rename(state = State,
         environment = Field.Location,
         genotype = Pedigree,
         earHeight = Ear.Height..cm.) %>%
  select(environment, state, genotype, earHeight)

g2f2022Meta <- read.csv('rawData/g2f_2022_field_metadata.csv') %>%
  rename(environment = Experiment_Code,
         city = City,
         lat = Latitude_of_Field_Corner_.1..lower.left.,
         lon = Longitude_of_Field_Corner_.1..lower.left.) %>%
  select(environment, city, lat, lon)

g2f2022 <- full_join(g2f2022, g2f2022Meta, join_by(environment), keep = FALSE, suffix = c('', ''), relationship = 'many-to-one') %>%
  mutate(source = 'G2F_2022')

phenotype_data <- bind_rows(panzea_data, g2f2014, g2f2015, g2f2020, g2f2021, g2f2022) %>%
                  rowwise() %>%
                  mutate(
                    genotype = str_to_upper(genotype),
                    environment = str_to_upper(environment)
                  ) %>%
                  filter(environment != '') %>%
                  mutate(
                    state = case_when(
                        state=='Colorado' ~ 'CO',
                        str_detect(environment, 'DE') ~ 'DE',
                        str_detect(environment, 'GA') ~ 'GA',
                        str_detect(environment, 'IA') ~ 'IA',
                        state=='Illinois' ~ 'IL',
                        state=='Indiana' ~ 'IN',
                        str_detect(environment, 'MI') ~ 'MI',
                        state=='Minnesota' ~ 'MN',
                        str_detect(environment, 'MO') ~ 'MO',
                        str_detect(environment, 'NC') ~ 'NC',
                        state=='North Carolina' ~ 'NC',
                        str_detect(environment, 'NE') ~ 'NE',
                        str_detect(environment, 'NY') ~ 'NY',
                        str_detect(environment, 'OH') ~ 'OH',
                        state=='South Carolina' ~ 'SC',
                        str_detect(environment, 'TX') ~ 'TX',
                        str_detect(environment, 'WI') ~ 'WI',
                        state=='Wisconsin' ~ 'WI',
                        .default = state
                    ),
                    city = str_split_i(city, ',', 1) %>%
                           str_split_i(';', 1) %>%
                           str_split_i('-', 1) %>%
                           case_when(.=='' ~ NA, .default = .)
                  ) %>%
                  filter((!is.na(lat) & !is.na(lon)) | (!is.na(city) & !is.na(state))) %>%
                  mutate(sourceEnvironment = str_c(source, environment, sep = ':')) %>%
                  filter(environment!='GEH1')

environment_data <- phenotype_data %>%
                    group_by(sourceEnvironment) %>%
                    summarise(
                        city = max(city, na.rm = TRUE),
                        state = max(state, na.rm = TRUE),
                        lat = mean(lat, na.rm = TRUE),
                        lon = mean(lon, na.rm = TRUE),
                        source = max(source, na.rm = TRUE),
                        environment = max(environment, na.rm = TRUE)
                    )


missing_lat_lon <- filter(environment_data, is.na(lat) & is.na(lon))
coords <- apply(missing_lat_lon, 1, function(x){google_geocode(address = paste(x['city'], x['state'], sep = ', '), key = 'AIzaSyBx0dJsUYphK3y144m_ldAOhc7loeHZUOo')})
missing_lat_lon <- cbind(missing_lat_lon, do.call(rbind, lapply(coords, geocode_coordinates)))

missing_lat_lon <- missing_lat_lon[, c(1:3, 6:9)] %>%
  rename(lat = lat,
         lon = lng) %>%
  select(environment, source, city, state, lat, lon, sourceEnvironment)

environment_coords <- filter(environment_data, !is.na(lat) & !is.na(lon))
environment_coords <- bind_rows(environment_coords, missing_lat_lon)

environment_data <- download_daymet(site = environment_coords$sourceEnvironment[1], lat = environment_coords$lat[1], lon = environment_coords$lon[1], start = 1980)
environment_data <- environment_data$data %>%
  mutate(sourceEnvironment = environment_coords$sourceEnvironment[1])

for (i in 2:length(environment_coords$sourceEnvironment))
{
  env_df <- download_daymet(site = environment_coords$sourceEnvironment[i],
                            lat = environment_coords$lat[i],
                            lon = environment_coords$lon[i],
                            start = 1980)
  env_df <- env_df$data %>% mutate(sourceEnvironment = environment_coords$sourceEnvironment[i])

  environment_data <- bind_rows(environment_data, env_df)
}


genotype_list <- read.csv('rawData/genotypeList.tsv', sep = '\t')[17:1531, ]
genotype_list <- tibble(genotypeVCF = genotype_list)
genotype_list <- genotype_list %>%
                 rowwise() %>%
                 mutate(genotypePheno = str_to_upper(genotypeVCF))

phenotype_data <- full_join(phenotype_Data, genotype_list, join_by(genotype==genotypePheno), keep = FALSE, suffix = c('', ''), relationship = 'many-to-one')

# Write Data ---------------------------------------------------

write.csv(phenotype_data, 'earHeight.csv')
write.csv(environment_data, 'weatherData.csv')
