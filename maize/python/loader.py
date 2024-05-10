"""
loader.py
Ian Kollipara <ikollipara2@huskers.unl.edu>

Load predefined and trained models for Maize-ML.
"""

# Imports
from pathlib import Path
from typing import Literal
import tensorflow as tf

model_dir = Path(__file__).parent / "_predefined"

Models = Literal["ear_height_predictor_sp89", "genotype_ae_89", "genotype_encoder_89", "weather_ae_97"]

def load_model(model: Models) -> tf.keras.Model:
    """ Load a predefined model from Maize-ML.

    The models were trained during the development of Maize-ML.

    Args:
        model (Literal["ear_height_predictor_sp89", "genotype_ae_89", "genotype_encoder_89", "weather_ae_97"]): The model to load.

    Returns:
        tf.keras.Model: The loaded model.
    """

    if model not in ["ear_height_predictor_sp89", "genotype_ae_89", "genotype_encoder_89", "weather_ae_97"]:
        raise ValueError("Invalid model name.")

    return tf.keras.models.load_model(model_dir / f"{model}.keras")
