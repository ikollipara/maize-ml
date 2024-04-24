"""
types.py
Ian Kollipara <ikollipara2@huskers.unl.edu>

Helper types for the project.
"""

from typing import Callable
import tensorflow as tf

__all__ = ['LossFunction']

LossFunction = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
