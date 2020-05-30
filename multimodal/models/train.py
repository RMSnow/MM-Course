# -*- coding: utf-8 -*-
""" 
@author: RMSnow 
@file: train.py 
@time: 2020/4/30 20:31
@contact: xueyao_98@foxmail.com

# train and visualization
"""
import numpy as np

from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.utils import class_weight

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
          multi_output=False, monitor=None, monitor_mode='auto',
          labels_name=None, use_class_weights=False, to_predict=True):
    model_file = './model/{}.hdf5'.format(model_name)

    if monitor is None:
        if not multi_output:
            monitor = 'val_acc'
        else:
            monitor = 'val_fake_acc'

    checkpoint = ModelCheckpoint(model_file, monitor=monitor, verbose=0,
                                 save_best_only=True, mode=monitor_mode, save_weights_only=True)
    if early_stop:
        early_stop = EarlyStopping(
            monitor=monitor, patience=10, verbose=1, mode=monitor_mode)
    else:
        early_stop = EarlyStopping(
            monitor=monitor, patience=epochs, verbose=1, mode=monitor_mode)

    print(model.summary())
    print()
    if not use_class_weights:
        model_history = model.fit(train_data, train_label, epochs=epochs,
                                  batch_size=batch_size, validation_data=(test_data, test_label),
                                  shuffle=True, callbacks=[checkpoint, early_stop])
    else:
        train_label_ints = [y.argmax() for y in train_label]
        cw = class_weight.compute_class_weight('balanced', np.unique(
            train_label_ints), train_label_ints)

        model_history = model.fit(train_data, train_label, epochs=epochs,
                                  batch_size=batch_size, validation_data=(test_data, test_label),
                                  shuffle=True, callbacks=[checkpoint, early_stop], class_weight=cw)

    loss_plot(model_history.history, multi_output)

    if to_predict:
        if error_analysis:
            predict_error_analysis(model, model_file, model_name, test_data, test_label)
        else:
            predict(model, model_file, test_data, test_label, multi_output, labels_name)


def predict_single_output(y_pred, test_label, labels_names):
    if type(test_label) == list and len(test_label) == 1:
        test_label = test_label[0]

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


def predict(model, model_file, test_data, test_label, multi_output=False, labels_name=None):
    model.load_weights(model_file)
    y_pred = model.predict(test_data)

    if not multi_output:
        if labels_name is None:
            predict_single_output(y_pred, test_label, labels_names=['truth', 'rumor'])
        else:
            predict_single_output(y_pred, test_label, labels_names=labels_name)
    else:
        assert len(y_pred) == 2
        predict_single_output(y_pred[0], test_label[0], labels_names=['truth', 'rumor'])
        predict_single_output(y_pred[1], test_label[1], labels_names=[
            '医药健康', '教育考试', '文体娱乐', '社会生活', '科技', '财经商业', '军事', '政治'])


def predict_error_analysis(model, model_file, model_name, test_data, test_label):
    pass
