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
from keras.layers import Reshape
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.initializers import Constant
from keras import backend as K
from keras.layers.core import Lambda

from FusionLayers import SelfAttLayer
from Config import related_branch_layer, unrelated_branch_layer, we_emb_matrix


class TwoBranchesBiGRU:
    def __init__(self, max_sequence_length=120, fusion_mode='attention',
                 use_related_branch=False, use_unrelated_branch=True,
                 hidden_units=32, output=2, l2_param=0.01, lr_param=0.001):
        self.max_sequence_length = max_sequence_length
        self.embedding_matrix = we_emb_matrix
        self.fusion_mode = fusion_mode
        self.use_related_branch = use_related_branch
        self.use_unrelated_branch = use_unrelated_branch

        self.hidden_units = hidden_units
        self.output = output
        self.l2_param = l2_param

        self.model = self.build()

        if use_related_branch:
            self.model.get_layer('related_branch').set_weights(related_branch_layer.get_weights())
        if use_unrelated_branch:
            self.model.get_layer('unrelated_branch').set_weights(unrelated_branch_layer.get_weights())

        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_param, beta_1=0.8),
                           metrics=['accuracy'])

    def build(self):
        semantic_input = Input(shape=(self.max_sequence_length,), name='Word2Vec')

        semantic_emb = Embedding(self.embedding_matrix.shape[0],
                                 self.embedding_matrix.shape[1],
                                 embeddings_initializer=Constant(self.embedding_matrix),
                                 input_length=self.max_sequence_length,
                                 trainable=False, name='Embedding')(semantic_input)
        # [n, steps, 64]
        gru = Bidirectional(GRU(self.hidden_units, return_sequences=True), name='BiGRU')(semantic_emb)
        avg_pool = GlobalAveragePooling1D(name='mean_pooling')(gru)
        max_pool = GlobalMaxPooling1D(name='max_pooling')(gru)
        # [n, 128]
        pool = Concatenate(name='gru_pooling')([avg_pool, max_pool])

        related_branch = Dense(32, activation='relu', trainable=False, name='related_branch')(pool)
        unrelated_branch = Dense(32, activation='relu', trainable=False, name='unrelated_branch')(pool)

        if self.use_related_branch and self.use_unrelated_branch:
            if self.fusion_mode == 'concat':
                # [n, 64]
                dense = Concatenate(name='two_branches_concat')([related_branch, unrelated_branch])
            elif self.fusion_mode == 'bilinear':
                # [n, 32, 32] -> [n, 32*32]
                bilinear = Lambda(self.bilinear_dot, name='two_branches_bilinear')([related_branch, unrelated_branch])
                flatten = Flatten()(bilinear)
                dense = Dropout(0.5)(flatten)
            elif self.fusion_mode == 'attention':
                # [n, 32] -> [n, 1, 32]
                related_branch_reshape = Reshape([1, 32])(related_branch)
                unrelated_branch_reshape = Reshape([1, 32])(unrelated_branch)
                branches = Concatenate(axis=1)([related_branch_reshape, unrelated_branch_reshape])
                dense = SelfAttLayer(name='two_branches_attention')(branches)
        elif self.use_unrelated_branch:
            dense = unrelated_branch
        elif self.use_related_branch:
            dense = related_branch
        else:
            print('至少使用一个分支！')
            return None

        output = Dense(self.output, activation='softmax', kernel_regularizer=l2(self.l2_param), name='fake_output')(
            dense)

        model = Model(inputs=[semantic_input], outputs=output)
        return model

    def bilinear_dot(self, branches):
        # branches[0],[1] : [n, 32] -> [n, 32, 1]
        a = K.expand_dims(branches[0])
        b = K.expand_dims(branches[1])
        # [n, 32, 32]
        dot = K.batch_dot(a, b, axes=[2, 2])

        sign_sqr = K.sign(dot) * K.sqrt(K.abs(dot) + 1e-10)
        return K.l2_normalize(sign_sqr, axis=-1)
