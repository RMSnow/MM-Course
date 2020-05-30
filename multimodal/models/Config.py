# -*- coding: utf-8 -*-
""" 
@author: RMSnow 
@file: Config.py 
@time: 2020/5/30 16:52
@contact: xueyao_98@foxmail.com

# 配置文件：引入外部参数，如 related branch/unrelated branch 等
"""

from keras.layers import Input
from keras.layers import Bidirectional
from keras.layers import GRU
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Dense
from keras.layers import Embedding, Concatenate
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.initializers import Constant

from GradientReversal import GradientReversal
import numpy as np


class BiGRU:
    def __init__(self, max_sequence_length, embedding_matrix, hidden_units=32,
                 output=2, gradient_reversal=False, l2_param=0.01, lr_param=0.001):
        self.max_sequence_length = max_sequence_length
        self.embedding_matrix = embedding_matrix
        self.hidden_units = hidden_units
        self.output = output
        self.gradient_reversal = gradient_reversal
        self.l2_param = l2_param

        self.model = self.build()
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_param, beta_1=0.8),
                           metrics=['accuracy'])

    def build(self):
        semantic_input = Input(shape=(self.max_sequence_length,), name='Word2Vec')

        semantic_emb = Embedding(self.embedding_matrix.shape[0],
                                 self.embedding_matrix.shape[1],
                                 embeddings_initializer=Constant(self.embedding_matrix),
                                 input_length=self.max_sequence_length,
                                 trainable=False)(semantic_input)

        gru = Bidirectional(GRU(self.hidden_units, return_sequences=True))(semantic_emb)
        avg_pool = GlobalAveragePooling1D()(gru)
        max_pool = GlobalMaxPooling1D()(gru)
        pool = Concatenate()([avg_pool, max_pool])

        dense = Dense(32, activation='relu',
                      kernel_regularizer=l2(self.l2_param), name='category_branch')(pool)
        if self.gradient_reversal:
            dense = GradientReversal(hp_lambda=1)(dense)
        output = Dense(self.output, activation='softmax', kernel_regularizer=l2(self.l2_param))(dense)

        model = Model(inputs=[semantic_input], outputs=output)
        return model


we_emb_matrix = np.load(
    '../data/we_embedding_matrix_(6000, 300).npy')
related_model = BiGRU(max_sequence_length=120, embedding_matrix=we_emb_matrix,
                      output=3, gradient_reversal=False).model
unrelated_model = BiGRU(max_sequence_length=120, embedding_matrix=we_emb_matrix,
                        output=3, gradient_reversal=True).model

related_model.load_weights(
    './model/Branches_Balanced3_BiGRU_RelatedBranch_useClassWeight.hdf5')
unrelated_model.load_weights(
    './model/Branches_Balanced3_BiGRU_UnrelatedBranch_useClassWeight.hdf5')

related_branch_layer = related_model.get_layer('category_branch')
unrelated_branch_layer = unrelated_model.get_layer('category_branch')
