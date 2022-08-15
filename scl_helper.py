#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 17:18:39 2022

@author: kosmas
"""
from tensorflow import keras
import tensorflow as tf 
import tensorflow_addons as tfa

#From: https://keras.io/examples/vision/supervised-contrastive-learning/
def add_probe(input_shape,encoder):
    """
    Parameters
    ----------
    input_shape : touple
        The shape of a sample
    encoder: keras model
        The encoder model
    """
    #Freeze the encoder
    for layer in encoder.layers:
        layer.trainable=False
    #create the new model
    inputs = keras.Input(shape=input_shape)
    features = encoder(inputs)
    outputs= keras.layers.Dense(2)(features)
    model = keras.Model(inputs=inputs,outputs=outputs)
    return model


#From: https://keras.io/examples/vision/supervised-contrastive-learning/
class SupervisedContrastiveLoss(keras.losses.Loss):
    """
   A class used to compute the loss function

   """
   
    def __init__(self, temperature=1, name=None):
        """
        Parameters
        ----------
        name : str
            The name pointer of the loss class
        temperature: float
            Influences the probability distribution over classes
        """
        
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors,sample_weight=None):
        """
        Parameters
        ----------
        labels : array
            The true labels of the dataset
        feature_vectors: array
            The samples of the dataset
        sample_weight: NoneType
            A reduction weighting coefficient for the per-sample losses 
            (set for cpmpatibility with the framework)
            
        """
        
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)

