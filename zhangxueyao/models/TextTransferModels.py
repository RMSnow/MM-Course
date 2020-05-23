# -*- coding: utf-8 -*-
""" 
@author: RMSnow 
@file: TextTransferModels.py 
@time: 2020/5/23 18:57
@contact: xueyao_98@foxmail.com

# Text transfer learning model
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


# EANN/DANN end2end model
class End2endBiGRU:
    def __init__(self, max_sequence_length, embedding_matrix, hidden_units=32, lambda_reversal_strength=1,
                 l2_param=0.01, lr_param=0.001):
        self.max_sequence_length = max_sequence_length
        self.embedding_matrix = embedding_matrix
        self.lambda_reversal_strength = lambda_reversal_strength
        self.hidden_units = hidden_units
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

        dense = Dense(32, activation='relu', kernel_regularizer=l2(self.l2_param))(pool)
        output = Dense(2, activation='softmax', kernel_regularizer=l2(self.l2_param), name='fake')(dense)

        category_in = GradientReversal(hp_lambda=self.lambda_reversal_strength)(pool)
        category_out = Dense(8, activation='softmax', kernel_regularizer=l2(self.l2_param), name='category')(
            category_in)

        model = Model(inputs=[semantic_input], outputs=[output, category_out])
        return model


class End2endTextCNN:
    def __init__(self, max_sequence_length, embedding_matrix, filters_num=100, lambda_reversal_strength=1,
                 l2_param=0.01, lr_param=0.001):
        self.max_sequence_length = max_sequence_length
        self.embedding_matrix = embedding_matrix
        self.filters_num = filters_num
        self.lambda_reversal_strength = lambda_reversal_strength
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

        dense = Dense(32, activation='relu', kernel_regularizer=l2(self.l2_param))(cnn)
        output = Dense(2, activation='softmax', kernel_regularizer=l2(self.l2_param), name='fake')(dense)

        category_in = GradientReversal(hp_lambda=self.lambda_reversal_strength)(cnn)
        category_out = Dense(8, activation='softmax', kernel_regularizer=l2(self.l2_param), name='category')(
            category_in)

        model = Model(inputs=[semantic_input], outputs=[output, category_out])
        return model
