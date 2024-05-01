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

__all__ = ['WeatherAutoencoder']

# Model Definitions

@tf.keras.utils.register_keras_serializable(package='MaizeML')
class WeatherAutoencoder(tf.keras.Model):
    """ WeatherAutoencoder.

    This class defines a recurrent autoencoder model for weather data.
    """

    def __init__(self, codings_size: int, n_steps: int, n_features: int, **kwargs):
        """ Initialize the WeatherAutoencoder.

        Args:
            codings_size (int): The size of the latent space codings.
            n_steps (int): The number of time steps in the input data.
            n_features (int): The number of features in the input data.
        """
        super().__init__(**kwargs)
        self.codings_size = codings_size
        self.n_steps = n_steps
        self.n_features = n_features

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    def _build_encoder(self) -> tf.keras.Model:
        """ Build the encoder.

        The encoder is a recurrent neural network with a GRU layer.
        """

        _encoder = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.n_steps, self.n_features)),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128)),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer='l1_l2'),
            tf.keras.layers.Dense(self.codings_size, activation='sigmoid')
        ])

        return _encoder

    def _build_decoder(self) -> tf.keras.Model:
        """ Build the decoder.

        The decoder is a recurrent neural network with a GRU layer.
        """

        _decoder = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.codings_size,)),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer='l1_l2'),
            tf.keras.layers.RepeatVector(self.n_steps),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True)),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.n_features, activation='sigmoid'))
        ])

        return _decoder

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """ Call the autoencoder.

        This method calls the autoencoder on the input tensor.
        """

        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_config(self) -> dict[str, Union[int, str]]:
        """ Get the configuration.

        This method gets the configuration of the WeatherAutoencoder.
        """

        config = super().get_config()
        config.update({
            'codings_size': self.codings_size,
            'n_steps': self.n_steps,
            'n_features': self.n_features
        })

        return config


    @staticmethod
    def normalize(data: tf.Tensor) -> tf.Tensor:
        """ Normalize the input data.

        This method normalizes the input data to $(0, 1]$.
        """

        max_tensor = tf.math.reduce_max(data, axis=0)
        return tf.math.divide_no_nan(data, max_tensor)
