"""
genotype_autoencoder.py
Ian Kollipara <ikollipara2@huskers.unl.edu>

This file contains the GenotypeAutoencoder class, which defines a
recurrent autoencoder model for genotype data.
"""

# Imports
from maize.python.types import LossFunction
import tensorflow as tf
import numpy as np
from typing import Optional, Union, Self

__all__ = ['GenotypeAutoencoder']

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

class GenotypeAutoencoder:
    """ GenotypeAutoencoder.

    This class defines a recurrent autoencoder model for genotype data.
    This class itself is a wrapper around a Keras model to provide a
    fluent interface for training and evaluation.
    """

    def __init__(self, codings_size: int, n_features: int) -> None:
        """ Initialize the GenotypeAutoencoder.

        Args:
            codings_size (int): The size of the latent space codings.
        """
        self.codings_size = codings_size
        self.n_features = n_features
        self.lookup = tf.keras.layers.StringLookup(vocabulary=[b'0/0', b'0/1', b'1/1', b'./.'], mask_token=None)

        self.model, self.encoder, self.decoder = self._build_model()

    def _build_model(self) -> tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
        """ Build the model.

        This method constructs the Keras model for the autoencoder.
        It defines a recurrent neural network with a GRU encoder and decoder.
        This does not compile the models.

        Returns:
            tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]: The model,
                encoder, and decoder models.
        """

        input_layer = tf.keras.Input(shape=(None, self.n_features), ragged=True)
        bidirectional_1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True))(input_layer)
        bidirectional_2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128))(bidirectional_1)
        dropout = tf.keras.layers.Dropout(0.3)(bidirectional_2)
        dense_1 = tf.keras.layers.Dense(256, activation='relu')(dropout)
        dense_2 = tf.keras.layers.Dense(self.codings_size, activation='sigmoid')(dense_1)

        _encoder = tf.keras.Model(inputs=input_layer, outputs=dense_2)

        latent_input = tf.keras.Input(shape=(self.codings_size,))
        dense_3 = tf.keras.layers.Dense(256, activation='relu')(latent_input)
        repeat_layer = tf.keras.layers.Lambda(_repeat)([dense_3, input_layer])
        bidirectional_4 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True))(repeat_layer)
        bidirectional_5 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True))(bidirectional_4)
        time_distributed = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.n_features, activation='softmax'))(bidirectional_5)

        _decoder = tf.keras.Model(inputs=latent_input, outputs=time_distributed)

        _model = tf.keras.Model(inputs=input_layer, outputs=_decoder(_encoder(input_layer)))

        return _model, _encoder, _decoder

    def _one_hot_encode(self, x: tf.Tensor) -> tf.RaggedTensor:
        """ One-hot encode the input tensor from a Genotype string to
        a ragged tensor.

        Args:
            x (tf.Tensor): The input tensor.

        Returns:
            tf.RaggedTensor: The one-hot encoded tensor.
        """

        x = tf.strings.split(x, ' ')
        x = self.lookup(x)
        x = tf.one_hot(x, depth=len(self.lookup.get_vocabulary()))
        return x

    def _calculate_accuracy(self, y_true: tf.RaggedTensor, y_pred: tf.RaggedTensor) -> tf.Tensor:
        accuracy = 0
        for i in range(len(y_true)):
            accuracy += tf.reduce_sum(tf.cast(tf.equal(y_true[i], y_pred[i]), tf.float32))

        return accuracy / (len(y_true) * y_true.row_lengths()[0])

    def compile(self, loss: Union[tf.keras.losses.Loss, LossFunction], optimizer: tf.keras.optimizers.Optimizer) -> Self:
        """ Compile the model.

        This method compiles the model with the given optimizer, loss function,
        and metrics.

        Args:
            optimizer (Union[str, tf.keras.optimizers.Optimizer]): The optimizer to use.
            loss (Union[str, tf.keras.losses.Loss]): The loss function to use.
            metrics (Optional[Iterable[Union[str, tf.keras.metrics.Metric]]]): The metrics to use.
        """
        self.optimizer = optimizer
        self.loss = loss

        return self

    def fit(
            self,
            training_dataset: tf.data.Dataset,
            *,
            validation_dataset: Optional[tf.data.Dataset] = None,
            epochs: int = 1,
            early_stopping: bool = True,
            patience: int = 10
        ) -> None:
        """ Fit the model.

        This method fits the model to the given training data.

        Args:
            training_dataset (tf.data.Dataset): The training dataset.
            validation_dataset (tf.data.Dataset): The validation dataset.
            epochs (int): The number of epochs to train for.
            verbose (int): The verbosity level.
        """

        history = {'loss': [], 'accuracy': []}
        if validation_dataset is not None:
            history['val_loss'] = []
            history['val_accuracy'] = []
        current_patience = 0
        min_val_loss = None

        for epoch in range(epochs):
            for X_batch in training_dataset:
                X_batch = self._one_hot_encode(X_batch)
                with tf.GradientTape() as tape:
                    y = self.model(X_batch)
                    loss = self.loss(X_batch, y)
                    gradients = tape.gradient(loss, self.model.trainable_variables)
                    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                scalar_loss = tf.reduce_mean(loss)
                accuracy = self._calculate_accuracy(X_batch, y)
                print(f'Epoch {epoch + 1} | Loss: {scalar_loss:.5}, Accuracy: {accuracy:.5}', end="\r")
            history['loss'].append(scalar_loss)
            history['accuracy'].append(accuracy)
            print(f"Epoch {epoch + 1} | Loss: {scalar_loss:.5}, Accuracy: {accuracy:.5}", end=" - ")
            if validation_dataset:
                val_loss = []
                val_accuracy = []
                for V_batch in validation_dataset:
                    V_batch = self._one_hot_encode(V_batch)
                    y = self.model(V_batch)
                    loss = self.loss(V_batch, y)
                    scalar_loss = tf.reduce_mean(loss)
                    accuracy = self._calculate_accuracy(V_batch, y)
                    val_loss.append(scalar_loss)
                    val_accuracy.append(accuracy)
                val_loss = tf.reduce_mean(val_loss)
                val_accuracy = tf.reduce_mean(val_accuracy)
                if min_val_loss is None or val_loss < min_val_loss:
                    min_val_loss = val_loss
                    current_patience = 0
                else:
                    current_patience += 1
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                print(f"Val Loss: {val_loss:.5}, Val Accuracy: {val_accuracy:.5}")
                if early_stopping and current_patience >= patience:
                    print(f"Early stopping after {epoch + 1} epochs.")
                    break

        self.history = history
        return self

    def evaluate(self, X: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """ Evaluate the model.

        This method evaluates the model on the given data.

        Args:
            X (tf.Tensor): The input data.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: The loss and accuracy.
        """
        test_accuracy = []
        test_loss = []
        for T_batch in X:
            T_batch = self._one_hot_encode(T_batch)
            y = self.model(T_batch)
            loss = self.loss(T_batch, y)
            scalar_loss = tf.reduce_mean(loss)
            accuracy = self._calculate_accuracy(T_batch, y)
            test_loss.append(scalar_loss)
            test_accuracy.append(accuracy)
        test_loss = tf.reduce_mean(test_loss)
        test_accuracy = tf.reduce_mean(test_accuracy)
        return test_loss, test_accuracy
