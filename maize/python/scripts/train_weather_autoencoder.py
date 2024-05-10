"""
train_weather_autoencoder.py
Ian Kollipara <ikollipara2@huskers.unl.edu>

The script to train the weather autoencoder for Maize-ML.
"""

from sys import argv
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from maize.python.models.weather_autoencoder import WeatherAutoencoder


if len(argv) == 2 and argv[1] == "--help":
    print("Usage: python train_weather_autoencoder.py <input_dir> <output_path>")
    print("Trains the weather autoencoder for Maize-ML.")
    print("Arguments:")
    print("\t<input_dir>: The directory containing the weather data.")
    print("\t<output_path>: The path to save the trained model to.")
    exit(0)

if len(argv) < 3:
    print("Usage: python train_weather_autoencoder.py <input_dir> <output_path>")
    exit(1)

if len(argv) > 3:
    flags = argv[3:]
    if "--num-splits" in flags:
        num_splits = flags[flags.index("--num-splits") + 1]
    if "--embedding-size" in flags:
        embedding_size = flags[flags.index("--embedding-size") + 1]

else:
    num_splits = 4
    embedding_size = 150

input_dir = argv[1]
output_path = argv[2]
scores = {}

def preprocess(time_series_record):
    """ Normalize the time series record by dividing by the max value. """
    max_tensor = tf.math.reduce_max(time_series_record, axis=0)
    return tf.math.divide_no_nan(time_series_record, max_tensor)


# Create Training Data
X = tf.map_fn(
    preprocess,
    tf.constant([
        pd.read_csv(
            f"{input_dir}/{env}_weather.csv"
        ).drop(
            columns=['Unnamed: 0.1', 'Unnamed: 0', 'yday', 'year', 'period']
        ).to_numpy()
        for env in os.listdir(input_dir)
        if env.endswith('.csv')
    ], dtype=tf.float32)
).numpy()

kfold = KFold(n_splits=num_splits, shuffle=True)

for idx, (train, test) in enumerate(kfold.split(X, X)):
    X_train = tf.data.Dataset.from_tensor_slices(tf.constant(X[train]))
    X_test = tf.data.Dataset.from_tensor_slices(tf.constant(X[test]))
    X_train = tf.data.Dataset.zip((X_train, X_train))
    X_test = tf.data.Dataset.zip((X_test, X_test))

    weather_ae = WeatherAutoencoder(codings_size=embedding_size, n_steps=15_695, n_features=7)
    weather_ae.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    print('--- Training Weather Autoencoder ---')
    print(f'Training for Fold {idx + 1}...')

    history = weather_ae.fit(
        X_train.batch(11),
        epochs=15,
    )

    scores[f"model_{idx+1}"] = weather_ae.evaluate(X_test.batch(11))

    np.save(f"{output_path}/history_{idx+1}.npy", history.history)
    weather_ae.save(f"{output_path}/model_{idx+1}.keras")

pd.DataFrame.from_dict(scores).to_csv(f"{output_path}/scores.csv")
print("Finished training Weather Autoencoder.")
