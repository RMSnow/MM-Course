# -*- coding: utf-8 -*-
""" 
@author: RMSnow 
@file: FusionLayers.py 
@time: 2020/5/5 15:53
@contact: xueyao_98@foxmail.com

# Fusion Layers
"""

from keras.engine.topology import Layer
from keras import initializers
from keras import backend as K
import tensorflow as tf


class SelfAttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        # self.init = initializers.get('glorot_uniform')
        super(SelfAttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: (n, steps, dim)
        dim = input_shape[-1]
        self.W = self.add_weight(shape=(dim,),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name))

    def call(self, x, mask=None):
        # (n, steps, dim) dot (dim,) -> (n, steps)
        e = K.exp(K.tanh(K.sum(x * self.W, axis=-1)))

        # (n, steps) / (n, 1) -> (n, steps)
        a = e / K.expand_dims(K.sum(e, axis=1), axis=-1)

        # (n, steps, dim) * (n, steps, 1) -> (n, steps, dim)
        weighted_input = x * K.expand_dims(a, axis=-1)
        # print(weighted_input.shape)

        # (n, steps, dim) -> (n, dim)
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # (n, steps, dim) -> (n, dim)
        return (input_shape[0], input_shape[-1])
