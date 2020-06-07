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
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout,BatchNormalization
from keras.layers import Reshape,multiply
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.initializers import Constant
from keras import backend as K
from keras.layers.core import Lambda

from FusionLayers import SelfAttLayer
from Config import related_branch_layer, unrelated_branch_layer, we_emb_matrix

class Multimodel:
    def __init__(self, max_sequence_length=120, fusion_mode='attention',
                 only_text_branch=False, only_img_branch=False, img_text_branches=False,
                 img_embed_size=512, hidden_units=32, output=2, l2_param=0.01, lr_param=0.001):
        
        self.max_sequence_length = max_sequence_length
        self.embedding_matrix = we_emb_matrix
        self.img_embed_size = img_embed_size
        self.fusion_mode = fusion_mode

        self.hidden_units = hidden_units
        self.output = output
        self.l2_param = l2_param

        self.only_img_branch=only_img_branch
        self.only_text_branch=only_text_branch
        self.img_text_branches=img_text_branches
        
        self.model = self.build()
        if only_text_branch or img_text_branches:
            self.model.get_layer('related_branch').set_weights(related_branch_layer.get_weights())
            self.model.get_layer('unrelated_branch').set_weights(unrelated_branch_layer.get_weights())
            
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_param, beta_1=0.8),
                           metrics=['accuracy'])
    
    def build(self):
        semantic_input = Input(shape=(self.max_sequence_length,), name='input_txt')
        img_input = Input(shape=(self.img_embed_size, ), name='input_img')
        
        semantic_emb = Embedding(self.embedding_matrix.shape[0],
                                 self.embedding_matrix.shape[1],
                                 embeddings_initializer=Constant(self.embedding_matrix),
                                 input_length=self.max_sequence_length,
                                 trainable=False, name='Embedding')(semantic_input)

        gru = Bidirectional(GRU(self.hidden_units, return_sequences=True), name='BiGRU')(semantic_emb)
        avg_pool = GlobalAveragePooling1D(name='mean_pooling')(gru)
        max_pool = GlobalMaxPooling1D(name='max_pooling')(gru)
        pool = Concatenate(name='gru_pooling')([avg_pool, max_pool])

        related_branch = Dense(32, activation='relu', trainable=False, name='related_branch')(pool)
        unrelated_branch = Dense(32, activation='relu', trainable=False, name='unrelated_branch')(pool)

        related_branch_reshape = Reshape([1, 32])(related_branch)
        unrelated_branch_reshape = Reshape([1, 32])(unrelated_branch)
        branches = Concatenate(axis=1)([related_branch_reshape, unrelated_branch_reshape])
        text_fusion = SelfAttLayer(name='two_branches_attention')(branches)

        if self.only_img_branch:
            dense=img_input
        elif self.only_text_branch:
            dense=text_fusion
        elif self.img_text_branches:
            if self.fusion_mode == 'concat':
                dense = Concatenate(name='two_branches_concat')([img_input, text_fusion])
            elif self.fusion_mode == 'bilinear':
                bilinear = Lambda(self.bilinear_dot, name='two_branches_bilinear')([img_input, text_fusion])
                flatten = Flatten()(bilinear)
                dense = Dropout(0.5)(flatten)
            elif self.fusion_mode == 'fc':
                concate = Concatenate(name='two_branches_concat')([img_input, text_fusion])
                dense = Dense(128, activation='relu', trainable=True, name='img_text_fc')(concate)
            elif self.fusion_mode == 'attention':
                #print(img_input.shape) [?,512]
                #print(text_fusion.shape) [?,32]
                img_input_reshape = Reshape([1, 512])(img_input)
                text_fusion_reshape = Reshape([1, 32])(text_fusion)
                imgtext_branches = Concatenate(axis=2)([img_input_reshape, text_fusion_reshape])
                dense = SelfAttLayer(name='img_text_attention')(imgtext_branches)

        output = Dense(self.output, activation='softmax', kernel_regularizer=l2(self.l2_param), name='fake_output')(dense)
        model = Model(inputs=[semantic_input,img_input], outputs=output)

        return model
    
    def bilinear_dot(self, branches):
        # branches[0],[1] : [n, 32] -> [n, 32, 1]
        a = K.expand_dims(branches[0])
        b = K.expand_dims(branches[1])
        # [n, 32, 32]
        dot = K.batch_dot(a, b, axes=[2, 2])

        sign_sqr = K.sign(dot) * K.sqrt(K.abs(dot) + 1e-10)
        return K.l2_normalize(sign_sqr, axis=-1)
    
class Multimodel_emoatt:
    def __init__(self, max_sequence_length=120, fusion_mode='attention',emoatt=False,img_embed_size=512, hidden_units=32,
                 output=2,img_emotion_size=256,semantic_emotion_size=55,l2_param=0.01, lr_param=0.001):
        
        self.max_sequence_length = max_sequence_length
        self.embedding_matrix = we_emb_matrix
        self.img_embed_size = img_embed_size
        self.fusion_mode = fusion_mode
        self.emoatt=emoatt

        
        self.semantic_emotion_size = semantic_emotion_size
        self.img_emotion_size = img_emotion_size

        self.hidden_units = hidden_units
        self.output = output
        self.l2_param = l2_param

        self.model = self.build()
        
        self.model.get_layer('related_branch').set_weights(related_branch_layer.get_weights())
        self.model.get_layer('unrelated_branch').set_weights(unrelated_branch_layer.get_weights())

        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_param, beta_1=0.8),
                           metrics=['accuracy'])
    
    def build(self):
        semantic_input = Input(shape=(self.max_sequence_length,), name='input_txt')
        img_input = Input(shape=(self.img_embed_size, ), name='input_img')
        semantic_emotion = Input(shape=(self.semantic_emotion_size,), name='txt_emotion')
        img_emotion = Input(shape=(self.img_emotion_size, ), name='img_emotion')
        
        semantic_emb = Embedding(self.embedding_matrix.shape[0],
                                 self.embedding_matrix.shape[1],
                                 embeddings_initializer=Constant(self.embedding_matrix),
                                 input_length=self.max_sequence_length,
                                 trainable=False, name='Embedding')(semantic_input)

        gru = Bidirectional(GRU(self.hidden_units, return_sequences=True), name='BiGRU')(semantic_emb)
        avg_pool = GlobalAveragePooling1D(name='mean_pooling')(gru)
        max_pool = GlobalMaxPooling1D(name='max_pooling')(gru)
        pool = Concatenate(name='gru_pooling')([avg_pool, max_pool])

        related_branch = Dense(32, activation='relu', trainable=False, name='related_branch')(pool)
        unrelated_branch = Dense(32, activation='relu', trainable=False, name='unrelated_branch')(pool)

        related_branch_reshape = Reshape([1, 32])(related_branch)
        unrelated_branch_reshape = Reshape([1, 32])(unrelated_branch)
        branches = Concatenate(axis=1)([related_branch_reshape, unrelated_branch_reshape])
        text_fusion = SelfAttLayer(name='two_branches_attention')(branches)

        if self.fusion_mode == 'concat':
            fusion = Concatenate(name='two_branches_concat')([img_input, text_fusion])
        elif self.fusion_mode == 'bilinear':
            bilinear = Lambda(self.bilinear_dot, name='two_branches_bilinear')([img_input, text_fusion])
            flatten = Flatten()(bilinear)
            fusion = Dropout(0.5)(flatten)
        elif self.fusion_mode == 'fc':
            concate = Concatenate(name='two_branches_concat')([img_input, text_fusion])
            fusion = Dense(128, activation='relu', trainable=True, name='img_text_fc')(concate)
        elif self.fusion_mode == 'attention':
            #print(img_input.shape) [?,512]
            #print(text_fusion.shape) [?,32]
            img_input_reshape = Reshape([1, 512])(img_input)
            text_fusion_reshape = Reshape([1, 32])(text_fusion)
            imgtext_branches = Concatenate(axis=2)([img_input_reshape, text_fusion_reshape])
            fusion = SelfAttLayer(name='img_text_attention')(imgtext_branches)

        if self.emoatt:
            emo_concat = Concatenate(name='emo_concat')([semantic_emotion, img_emotion])
            att_w = Dense(2, activation='softmax', kernel_regularizer=l2(self.l2_param), name='emo_weight')(emo_concat) #[?,2]
            att_w_reshape = Reshape([2, 1])(att_w)
            print(att_w_reshape)
            
            # 这些包含索引的操作是没有layer,node对应的
#           att_w_txt=att_w_reshape[:,0,:]
#           att_w_img=att_w_reshape[:,1,:]
#           att_rep_img = Lambda(lambda x : K.repeat_elements(x, 512, axis=1),name='att_rep_img')(att_w_reshape[:,1,:])
            
            # 索引操作采用lambda函数
            att_w_txt=Lambda(lambda x: x[:,0,:]) (att_w_reshape)
            att_w_img=Lambda(lambda x: x[:,1,:]) (att_w_reshape)
            
            att_rep_txt = Lambda(lambda x : K.repeat_elements(x, 32, axis=1),name='att_rep_txt')(att_w_txt)
            att_rep_img = Lambda(lambda x : K.repeat_elements(x, 512, axis=1),name='att_rep_img')(att_w_img)

            fusion_img=Lambda(lambda x: x[:,:512]) (fusion)
            fusion_txt=Lambda(lambda x: x[:,512:]) (fusion)
            att_txt = multiply([att_rep_txt, fusion_txt]) 
            att_img = multiply([att_rep_img, fusion_img])

            emo_att = Concatenate(name='emo_attention')([att_txt, att_img])
            dense = BatchNormalization(name='batch_norm')(emo_att)
        else:
            dense=fusion

        
        
        output = Dense(self.output, activation='softmax', kernel_regularizer=l2(self.l2_param), name='fake_output')(dense)
        model = Model(inputs=[semantic_input,img_input,semantic_emotion,img_emotion], outputs=output)

        return model
    
    def bilinear_dot(self, branches):
        # branches[0],[1] : [n, 32] -> [n, 32, 1]
        a = K.expand_dims(branches[0])
        b = K.expand_dims(branches[1])
        # [n, 32, 32]
        dot = K.batch_dot(a, b, axes=[2, 2])

        sign_sqr = K.sign(dot) * K.sqrt(K.abs(dot) + 1e-10)
        return K.l2_normalize(sign_sqr, axis=-1)
    
    def emo_att(self, att):
        att_rep_txt= K.repeat_elements(att[:,0,:], 32, axis=1)
        att_rep_img= K.repeat_elements(att[:,1,:], 512, axis=1)
        print(att_rep_txt.shape)
        print(att_rep_img.shape)
        return att_rep_txt,att_rep_img
    
class OneDense:
    def __init__(self, max_sequence_length=128, output=2, l2_param=0.01, lr_param=0.001):
        
        self.max_sequence_length = max_sequence_length

        self.output = output
        self.l2_param = l2_param

        self.model = self.build()
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_param, beta_1=0.8),
                           metrics=['accuracy'])
    
    def build(self):
        dense = Input(shape=(self.max_sequence_length,), name='input_feature')
        output = Dense(self.output, activation='softmax', kernel_regularizer=l2(self.l2_param), name='fake_output')(dense)
        model = Model(inputs=[dense], outputs=output)

        return model