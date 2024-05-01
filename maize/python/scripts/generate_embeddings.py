"""
generate_embeddings.py
Ian Kollipara <ikollipara2@huskers.unl.edu>

Script for generating embeddings for Maize-ML.
This script is used to generate the embeddings used in training
the generator head of the model. The embeddings are generated from
the genotype and weather data using the GenotypeAutoencoder and WeatherAutoencoder.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from maize.python.models.genotype_autoencoder import GenotypeAutoencoder

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

if len(sys.argv) != 3:
    print("Usage: python generate_embeddings.py <config_file>")
    print("Usage: cat <config> | python generate_embeddings.py -")
    sys.exit(1)

# Read the configuration

config = json.load(sys.stdin if sys.argv[1] == '-' else open(sys.argv[1], 'r'))
# Load the configuration
genotype_autoencoder = config['genotype_autoencoder']
weather_autoencoder = config['weather_autoencoder']
weather_dir = config['weather_dir']
genotype_dir = config['genotype_dir']
output_dir = config['output_dir']
ear_height_file = config['ear_height_file']


# Load the genotype autoencoder
genotype_autoencoder = tf.keras.models.load_model(genotype_autoencoder)
genotype_encoder = genotype_autoencoder.encoder

# Load the weather autoencoder
weather_autoencoder = tf.keras.models.load_model(weather_autoencoder)
weather_encoder = weather_autoencoder.encoder

sys.stdout.write("Generating weather embeddings...\n")
weather_embeddings = {
    env.split('.csv')[0][20:].upper(): weather_encoder(
        tf.expand_dims(
            tf.constant(
                pd.read_csv(os.path.join(weather_dir, env))
                .drop(columns=['Unnamed: 0.1', 'Unnamed: 0', 'yday', 'year', 'period'])
                .to_numpy()),
            axis=0
            )
        )[0]
    for env in tqdm(os.listdir(weather_dir)) if env.endswith('.csv')
}

sys.stdout.write("Generating genotype embeddings...\n")
ear_height_df = pd.read_csv(ear_height_file)
ear_height_grouped = ear_height_df.groupby('sourceEnvironment')
ear_height_grouped_by_env = {
   k.upper(): np.unique([
       ear_height_df.loc[label]['genotype'] for label in labels
   ]) for k, labels in ear_height_grouped.groups.items()
}

environment_genotypes = np.unique(np.concatenate(list(ear_height_grouped_by_env.values())))

genotypes = {}
for filename in tqdm(os.listdir(genotype_dir)):
    if filename.endswith('.csv'):
        env = filename.split('.csv')[0][20:].upper()
        c = 0
        with open(os.path.join(genotype_dir, filename)) as f:
            for line in f:
                c += 1
                if c >= 11:
                    genotype, *alleles = line.rstrip('\n').split(',')
                    if genotype.upper() in environment_genotypes:
                        if genotype_autoencoder.n_features is not None:
                            allele_len = len(alleles)
                            if allele_len > genotype_autoencoder.n_features:
                                alleles = alleles[:genotype_autoencoder.n_features]
                            else:
                                alleles += ['./.' for _ in range(genotype_autoencoder.n_features - allele_len)]
                        alleles = tf.expand_dims(alleles, axis=0)
                        if genotype.upper() in genotypes.keys():
                            genotypes[genotype.upper()].append(genotype_encoder(alleles)[0])
                        else:
                            genotypes[genotype.upper()] = [genotype_encoder(alleles)[0]]

# Concat the embeddings with (env, genotype) keys
embeddings = {
    (env, genotype): tf.concat([weather_embeddings[env], genotype_embedding], axis=0)
    for genotype_embedding in genotype for env, genotype in genotype_embedding.keys()
}

# Combine the embeddings with the ear height data
data = []
for (env, genotype), embedding in tqdm(embeddings.items()):
    ear_heights = ear_height_df[(ear_height_df['sourceEnvironment'] == env) & (ear_height_df['genotype'] == genotype.upper())]['ear_height'].to_numpy()
    for ear_height in ear_heights:
        for embed in embedding:
            data.append((env, embed.numpy(), ear_height))

# Save the embeddings
output_path = Path(output_dir)

if not output_path.exists():
    output_path.mkdir(parents=True)

output_file = output_path / 'embeddings.csv'

pd.DataFrame.from_records(data, columns=['sourceEnvironment', 'embedding', 'ear_height']).to_csv(output_file)

sys.stdout.write(f"Saved embeddings to {output_file}\n")
