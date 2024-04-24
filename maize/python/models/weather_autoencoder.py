"""
weather_autoencoder.py
Ian Kollipara <ikollipara2@huskers.unl.edu>

This file contains the WeatherAutoencoder class, which defines a
recurrent autoencoder model for weather data.
"""

# Imports
import tensorflow as tf
import numpy as np
from typing import Iterable, Optional, Union, Self
from maize.python.types import LossFunction

__all__ = ['WeatherAutoencoder']

# Model Definitions

class WeatherAutoencoder:
    """ WeatherAutoencoder.

    This class defines a recurrent autoencoder model for weather data.
    This class itself is a wrapper around a Keras model to provide a
    fluent interface for training and evaluation.
    """

    def __init__(self, codings_size: int, n_steps: int, n_features: int):
        """ Initialize the WeatherAutoencoder.

        Args:
            codings_size (int): The size of the latent space codings.
            n_steps (int): The number of time steps in the input data.
            n_features (int): The number of features in the input data.
        """
        self.codings_size = codings_size
        self.n_steps = n_steps
        self.n_features = n_features

        self.model, self.encoder, self.decoder = self._build_model()

    def _build_model(self) -> tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
        """ Build the model.

        This method constructs the Keras model for the autoencoder.
        The autoencoder is a recurrent neural network with a GRU encoder
        and decoder. This does not compile the models.

        Returns:
            tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]: The model,
                encoder, and decoder models.
        """

        _encoder = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.n_steps, self.n_features)),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128)),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer='l1_l2'),
            tf.keras.layers.Dense(self.codings_size, activation='sigmoid')
        ])

        _decoder = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.codings_size,)),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer='l1_l2'),
            tf.keras.layers.RepeatVector(self.n_steps),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True)),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.n_features, activation='sigmoid'))
        ])

        _model = tf.keras.models.Sequential([_encoder, _decoder])

        return _model, _encoder, _decoder

    def compile(self, loss: Union[str, tf.keras.Loss, LossFunction], optimizer: Union[str, tf.keras.Optimizer], metrics: Iterable[str, tf.keras.Metric] = None, debug: bool = False) -> Self:
        """ Compile the autoencoder models.

        This only compiles the autoencoder model, not the encoder or decoder.
        These are trained together.
        """
        metrics = metrics or []

        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics, run_eagerly=debug)
        return self

    def fit(
            self,
            training_dataset: tf.data.Dataset,
            *,
            epochs: int = 1,
            callbacks: Iterable[tf.keras.callbacks.Callback] = None,
            validation_dataset: Optional[tf.data.Dataset] = None
        ) -> Self:
        """ Fit the autoencoder models.

        This fits the encoder and decoder models together.
        """
        self.history = self.model.fit(training_dataset, epochs=epochs, callbacks=callbacks or [], validation_data=validation_dataset).history
        return self

    def evaluate(self, testing_dataset: tf.data.Dataset) -> Self:
        """ Evaluate the autoencoder models.

        This evaluates the encoder and decoder models together.
        """
        self.evaluation_metrics = self.model.evaluate(testing_dataset)
        return self
