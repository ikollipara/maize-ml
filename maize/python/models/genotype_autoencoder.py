"""
genotype_autoencoder.py
Ian Kollipara <ikollipara2@huskers.unl.edu>

This file contains the GenotypeAutoencoder class, which defines a
recurrent autoencoder model for genotype data.
"""

# Imports
from maize.python.layers.repeat import Repeat
import tensorflow as tf
import numpy as np
from typing import Optional, Union, Self

__all__ = ['GenotypeAutoencoder']

@tf.keras.utils.register_keras_serializable(package='MaizeML')
class GenotypeAutoencoder(tf.keras.Model):
    """ GenotypeAutoencoder.

    This class defines a recurrent autoencoder model for genotype data.
    """

    def __init__(self, codings_size: int, n_features: Optional[int] = None, **kwargs):
        """ Initialize the GenotypeAutoencoder.

        Args:
            codings_size (int): The size of the latent space codings.
            n_features (Optional[int]): The number of features in the input data.
                                        If None, then the number of features is determined at runtime,
                                        and the model uses ragged tensors (experimental).
        """
        super().__init__(**kwargs)
        self.codings_size = codings_size
        self.n_features = n_features
        self.lookup = tf.keras.layers.StringLookup(vocabulary=[b'0/0', b'0/1', b'1/1', b'./.'])
        self._input = (
                tf.keras.layers.Input(
                    shape=(None, self.n_features, len(self.lookup.get_vocabulary())),
                    ragged=True
                )
                if n_features is None else
                tf.keras.layers.Input(
                    shape=(self.n_features,len(self.lookup.get_vocabulary()))
                )
            )

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    def _build_encoder(self) -> tf.keras.Model:
        """ Build the encoder.

        The encoder is a recurrent neural network with a GRU layer.
        """


        _encoder = tf.keras.models.Sequential([
            self._input,
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer='l1_l2'),
            tf.keras.layers.Dense(self.codings_size, activation='sigmoid')
        ])

        return _encoder

    def _build_decoder(self) -> tf.keras.Model:
        """ Build the decoder.

        The decoder is a recurrent neural network with a GRU layer.
        """

        _latent_input = tf.keras.Input(shape=(self.codings_size,))
        _dense_1 = tf.keras.layers.Dense(256, activation='relu')(_latent_input)
        _repeat_layer = Repeat()([_dense_1, self._input]) if self.n_features is None else tf.keras.layers.RepeatVector(self.n_features)(_dense_1)
        _bidirectional_1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True))(_repeat_layer)
        _bidirectional_2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True))(_bidirectional_1)
        _time_distributed = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(self.lookup.get_vocabulary()), activation='softmax'))(_bidirectional_2)
        _decoder = tf.keras.Model(inputs=_latent_input, outputs=_time_distributed)

        return _decoder

    def _one_hot_encode(self, x: tf.Tensor) -> Union[tf.Tensor, tf.RaggedTensor]:
        """ One-hot encode the input tensor from a Genotype string to a tensor.

        The returned tensor is ragged if the n_features is not specified.

        Args:
            x (tf.Tensor): The input tensor.

        Returns:
            tf.RaggedTensor: The one-hot encoded tensor if n_features is None.
            tf.Tensor: The one-hot encoded tensor if n_features is not None.
        """

        x = tf.strings.split(x, ' ')
        x = self.lookup(x)
        x = tf.one_hot(x, depth=len(self.lookup.get_vocabulary()))
        if self.n_features is not None:
            x = tf.reshape(x, (-1, self.n_features, len(self.lookup.get_vocabulary())))
        return x

    def call(self, inputs: Union[tf.Tensor, tf.RaggedTensor]) -> Union[tf.Tensor, tf.RaggedTensor]:
        """ Call.

        This method calls the GenotypeAutoencoder model on the input tensor.

        Args:
            inputs (Union[tf.Tensor, tf.RaggedTensor]): The input tensor.

        Returns:
            tf.Tensor: The output tensor.
        """
        return self.decoder(self.encoder(inputs))

    def train_step(self, data):
        x, y = data

        x = self._one_hot_encode(x)
        y = self._one_hot_encode(y)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(y, y_pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data

        x = self._one_hot_encode(x)
        y = self._one_hot_encode(y)

        y_pred = self(x, training=False)
        loss = self.compute_loss(y, y_pred)

        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    def get_config(self) -> dict[str, str]:
        """ Get the configuration.

        This method gets the configuration of the GenotypeAutoencoder model.

        Returns:
            dict[str, str]: The configuration of the model.
        """
        config = super().get_config()
        config.update({
            'codings_size': self.codings_size,
            'n_features': self.n_features
        })
        return config

    @classmethod
    def from_config(cls, config: dict[str, str]) -> Self:
        """ Create the model from the configuration.

        This method creates the GenotypeAutoencoder model from the configuration.

        Args:
            config (dict[str, str]): The configuration of the model.

        Returns:
            Self: The GenotypeAutoencoder model.
        """
        return cls(**config)
