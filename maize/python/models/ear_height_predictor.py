"""
ear_height_predictor.py
Sebastian Kyllmann <skyllmann2@huskers.unl.edu>
Ian Kollipara <ikollipara2@huskers.unl.edu>

CNN prediction head for Maize-ML.
"""

# Imports
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package='Maize-ML')
class EarHeightPredictor(tf.keras.Model):
    """ CNN prediction head for Maize-ML.

    This class defines a CNN based prediction head for Maize-ML. The prediction head
    takes in embeddings from the generator head and outputs a prediction, in particular
    this prediction head is used to predict the ear height of maize plants.
    """

    def __init__(self, **kwargs):
        super().__init__(self, **kwargs)
        self.conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')
        self.max_pool1 = tf.keras.layers.MaxPooling1D(pool_size=2)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(16, activation='relu')
        self.output = tf.keras.layers.Dense(1)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """ Forward pass through the prediction head.

        Args:
            x (tf.Tensor): The input tensor to the prediction head.
            training (bool): Whether the model is in training mode.

        Returns:
            tf.Tensor: The output of the prediction head.
        """
        x = self.conv1(x, training=training)
        x = self.max_pool1(x, training=training)
        x = self.flatten(x, training=training)
        x = self.dense1(x, training=training)
        x = self.dense2(x, training=training)
        return self.output(x, training=training)

    def get_config(self) -> dict:
        """ Get the configuration of the prediction head.

        Returns:
            dict: The configuration of the prediction head.
        """
        return super().get_config()

    @classmethod
    def from_config(cls, config: dict) -> 'EarHeightPredictor':
        """ Create a prediction head from a configuration.

        Args:
            config (dict): The configuration of the prediction head.

        Returns:
            EarHeightPredictor: The prediction head created from the configuration.
        """
        return cls(**config)
