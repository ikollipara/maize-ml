"""
train_genotype_autoencoder.py
Ian Kollipara <ikollipara2@huskers.unl.edu>

The script to train the genotype autoencoder for Maize-ML.
"""

# Imports
from sys import argv
import tensorflow as tf
import numpy as np
import pandas as pd

# Check for help flag
if len(argv) == 2 and argv[1] == "--help":
    print("Usage: python train_genotype_autoencoder.py <input_dir> <output_path>")
    print("Trains the genotype autoencoder for Maize-ML.")
    print("Arguments:")
    print("\t<input_dir>: The directory containing the genotype data.")
    print("\t<output_path>: The path to save the trained model to.")
    exit(0)

# Check for correct number of arguments
if len(argv) < 3:
    print("Usage: python train_genotype_autoencoder.py <input_dir> <output_path>")
    exit(1)

if len(argv) > 3:
    flags = argv[3:]
    if "--embedding-size" in flags:
        embedding_size = flags[flags.index("--embedding-size") + 1]
    if "--n-features" in flags:
        n_features = flags[flags.index("--n-features") + 1]
        if n_features == "None":
            n_features = None
else:
    embedding_size = 100
    n_features = 10_000


from maize.python.models.genotype_autoencoder import GenotypeAutoencoder

input_dir = argv[1]
output_path = argv[2]

# Load the data
train_ds = tf.data.TextLineDataset(f"{input_dir}/train.txt")
val_ds = tf.data.TextLineDataset(f"{input_dir}/val.txt")
test_ds = tf.data.TextLineDataset(f"{input_dir}/test.txt")

model = GenotypeAutoencoder(codings_size=embedding_size, n_features=n_features)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    tf.data.Dataset.zip(train_ds, train_ds).batch(32),
    epochs=100,
    validation_data=tf.data.Dataset.zip(val_ds, val_ds).batch(32),
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=15)]
)

scores = model.evaluate(tf.data.Dataset.zip(test_ds, test_ds).batch(32))

np.save(f"{output_path}/history.npy", history.history)
model.save(f"{output_path}/model.keras")
model.encoder.save(f"{output_path}/encoder.keras")
pd.DataFrame.from_dict(scores).to_csv(f"{output_path}/scores.csv")

print("Model trained and saved.")
