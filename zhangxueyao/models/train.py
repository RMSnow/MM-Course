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


def loss_plot(hist_logs, multi_output=False):
    if not multi_output:
        plt.plot(hist_logs['acc'], marker='*')
        plt.plot(hist_logs['val_acc'], marker='*')
        plt.plot(hist_logs['loss'], marker='*')
        plt.plot(hist_logs['val_loss'], marker='*')

        plt.title('model accuracy/loss')
        plt.ylabel('accuracy/loss')
        plt.xlabel('epoch')
        plt.legend(['train_acc', 'val_acc', 'train_loss', 'val_loss'], loc='upper left')
    else:
        fig, axes = plt.subplots(1, 2, sharey=True, figsize=(18, 4))

        axes[0].plot(hist_logs['loss'], marker='*', label='train_loss')
        axes[0].plot(hist_logs['fake_loss'], marker='*', label='train_fake_loss')
        axes[0].plot(hist_logs['fake_acc'], marker='*', label='train_fake_acc')
        axes[0].plot(hist_logs['val_loss'], marker='*', label='val_loss')
        axes[0].plot(hist_logs['val_fake_loss'], marker='*', label='val_fake_loss')
        axes[0].plot(hist_logs['val_fake_acc'], marker='*', label='val_fake_acc')

        axes[0].set_title('model accuracy/loss')
        axes[0].set_ylabel('accuracy/loss')
        axes[0].set_xlabel('epoch')
        axes[0].legend(loc="upper left", ncol=3)

        axes[1].plot(hist_logs['loss'], marker='*', label='train_loss')
        axes[1].plot(hist_logs['category_loss'], marker='*', label='train_category_loss')
        axes[1].plot(hist_logs['category_acc'], marker='*', label='train_category_acc')
        axes[1].plot(hist_logs['val_loss'], marker='*', label='val_loss')
        axes[1].plot(hist_logs['val_category_loss'], marker='*', label='val_category_loss')
        axes[1].plot(hist_logs['val_category_acc'], marker='*', label='val_category_acc')

        axes[1].set_title('model accuracy/loss')
        axes[1].set_ylabel('accuracy/loss')
        axes[1].set_xlabel('epoch')
        axes[1].legend(loc="upper left", ncol=3)

    plt.show()


def train(model, model_name, train_data, test_data, train_label, test_label,
          epochs=20, batch_size=128, early_stop=True, error_analysis=False,
          multi_output=False):
    model_file = './model/{}.hdf5'.format(model_name)

    if not multi_output:
        monitor = 'val_acc'
    else:
        monitor = 'val_fake_acc'

    checkpoint = ModelCheckpoint(model_file, monitor=monitor, verbose=0,
                                 save_best_only=True, mode='auto', save_weights_only=True)
    if early_stop:
        early_stop = EarlyStopping(
            monitor=monitor, patience=10, verbose=1, mode='auto')
    else:
        early_stop = EarlyStopping(
            monitor=monitor, patience=epochs, verbose=1, mode='auto')

    print(model.summary())
    print()
    model_history = model.fit(train_data, train_label, epochs=epochs,
                              batch_size=batch_size, validation_data=(test_data, test_label),
                              shuffle=True, callbacks=[checkpoint, early_stop])

    loss_plot(model_history.history, multi_output)

    if error_analysis:
        predict_error_analysis(model, model_file, model_name, test_data, test_label)
    else:
        predict(model, model_file, test_data, test_label, multi_output)


def predict_single_output(y_pred, test_label, labels_names):
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
    print(classification_report(test_label, y_pred_label,
                                labels=[i for i in range(len(labels_names))],
                                target_names=labels_names, digits=3))
    print()


def predict(model, model_file, test_data, test_label, multi_output=False):
    model.load_weights(model_file)
    y_pred = model.predict(test_data)

    if not multi_output:
        predict_single_output(y_pred, test_label, labels_names=['truth', 'rumor'])
    else:
        assert len(y_pred) == 2
        predict_single_output(y_pred[0], test_label[0], labels_names=['truth', 'rumor'])
        predict_single_output(y_pred[1], test_label[1], labels_names=[
            '医药健康', '教育考试', '文体娱乐', '社会生活', '科技', '财经商业', '军事', '政治'])


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
