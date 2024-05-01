"""
repeat.py
Ian Kollipara <ikollipara2@huskers.unl.edu>

This file contains the Repeat layer, which repeats the input tensor along the time axis.
The repeat function is used to repeat the input tensor by a dynamic amount, specified by the second element of the input.
The Repeat layer is used in the GenotypeAutoencoder model to repeat the input tensor along the time axis an amount
known only at runtime.
"""

# Imports
from typing import Self
import tensorflow as tf

__all__ = ['Repeat']

def _repeat(x: tuple[tf.Tensor, tf.RaggedTensor]) -> tf.RaggedTensor:
    """ Repeat the input tensor.

    This function repeats the input tensor along the time axis.
    This particular function will repeat the given tensor by a
    dynamic amount, specified by the second element of the input.

    Args:
        x (tuple[tf.Tensor, tf.RaggedTensor]): The input tensor.

    Returns:
        tf.RaggedTensor: The repeated tensor.
    """
    x, inp = x

    max_row_length = tf.math.reduce_max(inp.row_lengths())
    x = tf.expand_dims(x, axis=1)
    x = tf.repeat(x, repeats=max_row_length, axis=1)
    x = tf.RaggedTensor.from_tensor(x, inp.row_lengths())
    return x

class Repeat(tf.keras.layers.Layer):
    """ Repeat.

    This class defines a custom layer that repeats the input tensor along the time axis.
    This layer is used in the GenotypeAutoencoder model to repeat the input tensor along
    the time axis an amount known only at runtime.
    """

    def call(self, inputs: tuple[tf.Tensor, tf.RaggedTensor]) -> tf.RaggedTensor:
        """ Call.

        This method calls the Repeat layer on the input tensor.

        Args:
            inputs (tuple[tf.Tensor, tf.RaggedTensor]): The input tensor.

        Returns:
            tf.RaggedTensor: The repeated tensor.
        """
        return _repeat(inputs)

    def get_config(self) -> dict[str, str]:
        """ Get the configuration.

        This method gets the configuration of the Repeat layer.

        Returns:
            dict[str, str]: The configuration of the layer.
        """
        return {}

    @classmethod
    def from_config(cls, config: dict[str, str]) -> Self:
        """ Create the layer from the configuration.

        This method creates the Repeat layer from the configuration.

        Args:
            config (dict[str, str]): The configuration of the layer.

        Returns:
            Self: The Repeat layer.
        """
        return cls(**config)
