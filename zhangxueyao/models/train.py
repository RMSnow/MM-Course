# -*- coding: utf-8 -*-
""" 
@author: RMSnow 
@file: train.py 
@time: 2020/4/30 20:31
@contact: xueyao_98@foxmail.com

# train and visualization
"""
import numpy as np
import os

from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from keras.models import load_model
import joblib

import matplotlib.pyplot as plt


def loss_plot(hist_logs):
    plt.plot()

    # keras 版本问题
    try:
        plt.plot(hist_logs['acc'], marker='*')
    except KeyError:
        plt.plot(hist_logs['accuracy'], marker='*')
    try:
        plt.plot(hist_logs['val_acc'], marker='*')
    except KeyError:
        plt.plot(hist_logs['val_accuracy'], marker='*')
    plt.plot(hist_logs['loss'], marker='*')
    plt.plot(hist_logs['val_loss'], marker='*')

    plt.title('model accuracy/loss')
    plt.ylabel('accuracy/loss')
    plt.xlabel('epoch')
    plt.legend(['train_acc', 'val_acc', 'train_loss', 'val_loss'], loc='upper left')
    plt.show()


def train(model, model_name, train_data, test_data, train_label, test_label,
          epochs=20, batch_size=128, early_stop=True, error_analysis=False):
    model_file = './model/{}.hdf5'.format(model_name)
    checkpoint = ModelCheckpoint(model_file, monitor='val_acc', verbose=0,
                                 save_best_only=True, mode='auto', save_weights_only=True)
    if early_stop:
        early_stop = EarlyStopping(
            monitor='val_acc', patience=10, verbose=1, mode='auto')
    else:
        early_stop = EarlyStopping(
            monitor='val_acc', patience=epochs, verbose=1, mode='auto')

    model_history = model.fit(train_data, train_label, epochs=epochs,
                              batch_size=batch_size, validation_data=(test_data, test_label),
                              shuffle=True, callbacks=[checkpoint, early_stop])
    loss_plot(model_history.history)

    if error_analysis:
        predict_error_analysis(model, model_file, model_name, test_data, test_label)
    else:
        predict(model, model_file, test_data, test_label)


def predict(model, model_file, test_data, test_label):
    model.load_weights(model_file)
    y_pred = model.predict(test_data)

    arg = y_pred.argmax(axis=1)

    y_pred_label = np.zeros(y_pred.shape)
    for i, index in enumerate(arg):
        y_pred_label[i][index] = 1

    accuracy = accuracy_score(test_label, y_pred_label)

    print()
    print('TEST_sz:', len(test_label))
    # print('test: {}+, {}-'.format(int(sum(test_label)), int(len(test_label) - sum(test_label))))
    print()
    print('Accuracy: {}'.format(accuracy))
    print()
    print(classification_report(test_label, y_pred_label, labels=[0, 1],
                                target_names=['truth', 'rumor'], digits=3))
    print()

    return y_pred


def predict_surely_analysis(model, model_file, test_data, test_label, analysis_num=10):
    y_pred = predict(model, model_file, test_data, test_label)
    prob = y_pred.argsort()

    sure_and_right = []
    sure_but_wrong = []

    # 概率值从小到大
    for i in prob:
        if len(sure_but_wrong) == analysis_num:
            break
        if test_label[i] == 1:
            sure_but_wrong.append(i)
    for i in prob:
        if len(sure_and_right) == analysis_num:
            break
        if test_label[i] == 0:
            sure_and_right.append(i)

    # 概率值从大到小
    prob.reverse()
    for i in prob:
        if len(sure_but_wrong) == analysis_num * 2:
            break
        if test_label[i] == 0:
            sure_but_wrong.append(i)
    for i in prob:
        if len(sure_and_right) == analysis_num * 2:
            break
        if test_label[i] == 1:
            sure_and_right.append(i)

    return sure_and_right, sure_but_wrong


def predict_error_analysis(model, model_file, model_name, test_data, test_label):
    pass
