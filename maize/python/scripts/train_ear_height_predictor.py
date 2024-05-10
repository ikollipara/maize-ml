"""
train_ear_height_predictor.py
Sebastian Kyllmann <skyllmann2@huskers.unl.edu>
Ian Kollipara <ikollipara2@huskers.unl.edu>

The script to train the ear height predictor for Maize-ML.
This script relies on the embeddings being created through the
`generate_embeddings.py` script.
"""

# Imports
from sys import argv
import re

try:
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf
except ImportError as e:
    print("Error: Missing required package: ", re.findall(r"'([^']*)'", str(e))[0])
    exit(1)

if len(argv) != 3:
    print("Usage: python train_ear_height_predictor.py <input_dir> <output_path>")
    exit(1)

if len(argv) == 2 and argv[1] == "--help":
    print("Usage: python train_ear_height_predictor.py <input_dir> <output_path>")
    print("Trains the ear height predictor for Maize-ML.")
    print("Arguments:")
    print("\t<input_dir>: The directory containing the embeddings.")
    print("\t<output_path>: The path to save the trained model to.")
    exit(0)

def string_list_to_float_list(s):
    """ Process a stringified list of floats into real floats."""

    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", s)
    return [float(n) for n in numbers]

input_dir = argv[1]
output_path = argv[2]

# Load the embeddings
train_embeddings = pd.read_csv(f"{input_dir}/train.csv")
val_embeddings = pd.read_csv(f"{input_dir}/val.csv")
test_embeddings = pd.read_csv(f"{input_dir}/test.csv")

# Separate Data and Labels
train_data = train_embeddings['data']
train_labels = train_embeddings['label']
val_data = val_embeddings['data']
val_labels = val_embeddings['label']
test_data = test_embeddings['data']
test_labels = test_embeddings['label']

# Process the data
train_data = train_data.apply(string_list_to_float_list).to_numpy()
val_data = val_data.apply(string_list_to_float_list).to_numpy()
test_data = test_data.apply(string_list_to_float_list).to_numpy()

train_data = train_data.reshape(-1, train_data.shape[1], 1)
val_data = val_data.reshape(-1, val_data.shape[1], 1)
test_data = test_data.reshape(-1, test_data.shape[1], 1)

train_labels = train_labels.to_numpy().reshape(-1, 1)
val_labels = val_labels.to_numpy().reshape(-1, 1)
test_labels = test_labels.to_numpy().reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data.reshape(-1, train_data.shape[1])).reshape(-1, train_data.shape[1], 1)
val_data = scaler.transform(val_data.reshape(-1, val_data.shape[1])).reshape(-1, val_data.shape[1], 1)
test_data = scaler.transform(test_data.reshape(-1, test_data.shape[1])).reshape(-1, test_data.shape[1], 1)

# Train the model
from maize.python.models.ear_height_predictor import EarHeightPredictor

model = EarHeightPredictor()
model.compile(optimizer='rmsprop', loss=tf.keras.losses.Huber(), metrics=['mse', 'mae'])
history = model.fit(
    train_data,
    train_labels,
    validation_data=(val_data, val_labels),
    epochs=200,
    batch_size=32,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=20)]
)

# Evaluate the model
results = model.evaluate(test_data, test_labels)
test_predictions = model.predict(test_data)

spearman = (
    pd.DataFrame({ 'Actual': test_predictions.flatten(), 'Predicted': test_labels.flatten() })
    .corr(method="spearman")
    .iloc[0,1]
)
print(f"Spearman Correlation: {spearman}")

# Save the model
model.save(output_path)
np.save(f"{output_path}.npy", results)
np.save(f"{output_path}_history.npy", history.history)

print("==============================================")
print(f"Model saved to {output_path}")
print(f"Results: {results}")
print("==============================================")
