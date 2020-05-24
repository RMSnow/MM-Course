# -*- coding: utf-8 -*-
""" 
@author: RMSnow 
@file: SimpleModels.py 
@time: 2020/5/10 23:01
@contact: xueyao_98@foxmail.com

# Text Models
"""
from keras.layers import Input
from keras.layers import Bidirectional
from keras.layers import GRU
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Dense
from keras.layers import Embedding, Concatenate
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.initializers import Constant
from keras import backend as K
from keras.layers.core import Lambda

from GradientReversal import GradientReversal


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


class TextCNN:
    def __init__(self, max_sequence_length, embedding_matrix, filters_num=256,
                 output=2, gradient_reversal=False, l2_param=0.01, lr_param=0.001):
        self.max_sequence_length = max_sequence_length
        self.embedding_matrix = embedding_matrix
        self.filters_num = filters_num
        self.output = output
        self.gradient_reversal = gradient_reversal
        self.l2_param = l2_param

        self.model = self.build()
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_param, beta_1=0.8),
                           metrics=['accuracy'])

    def build(self):
        semantic_input = Input(shape=(self.max_sequence_length,), name='Word2Vec')

        # [n, steps, dim]
        semantic_emb = Embedding(self.embedding_matrix.shape[0],
                                 self.embedding_matrix.shape[1],
                                 embeddings_initializer=Constant(self.embedding_matrix),
                                 input_length=self.max_sequence_length,
                                 trainable=False)(semantic_input)

        convs = []
        filter_sizes = [3, 4, 5]

        for filter_size in filter_sizes:
            # [n, max_seq_len - flz + 1, filters_num]
            conv = Conv1D(filters=self.filters_num, kernel_size=filter_size, activation='relu')(semantic_emb)
            # [n, 1, filters_num]
            conv_pool = MaxPooling1D(self.max_sequence_length - filter_size + 1)(conv)
            # [n, filters_num]
            conv_flat = Flatten()(conv_pool)

            convs.append(conv_flat)

        cnn_merge = Concatenate()(convs)
        cnn = Dropout(0.5)(cnn_merge)

        dense = Dense(32, activation='relu',
                      kernel_regularizer=l2(self.l2_param), name='category_branch')(cnn)
        if self.gradient_reversal:
            dense = GradientReversal(hp_lambda=1)(dense)
        output = Dense(self.output, activation='softmax', kernel_regularizer=l2(self.l2_param))(dense)

        model = Model(inputs=[semantic_input], outputs=output)
        return model
