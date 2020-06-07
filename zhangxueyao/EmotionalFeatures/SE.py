# -*- coding: utf-8 -*-
""" 
@author: RMSnow 
@file: SE-ch.py 
@time: 2020/5/10 14:13
@contact: xueyao_98@foxmail.com

# 把 SE-ch.ipynb 的各个函数加载进来
"""

import joblib
import numpy as np
import pandas as pd

# ============================== 加载否定词 ==============================
negation_words = []
with open('./doc/negationWords.txt', 'r') as src:
    lines = src.readlines()
    for line in lines:
        negation_words.append(line.strip())

len(negation_words)

# ============================== 加载程度词 ==============================
how_words_dict = dict()
with open('./doc/Hownet/intensifierWords.txt', 'r') as src:
    lines = src.readlines()
    for line in lines:
        how_word = line.strip().split()
        if len(how_word) != 2:
            # print(line)
            continue
        else:
            how_words_dict[how_word[0]] = float(how_word[1])

len(how_words_dict)

how_words_dict['有点']

how_words = list(how_words_dict.keys())
len(how_words)

how_words[:5]


# ============================== Hownet ==============================

def init_words(file):
    with open(file, 'r', encoding='utf-8') as src:
        words = src.readlines()
        words = [l.strip() for l in words]
    print('File: {}, Words_sz = {}'.format(file.split('/')[-1], len(words)))
    return words


pos_words = init_words('./doc/Hownet/正面情感词语（中文）.txt')
pos_words += init_words('./doc/Hownet/正面评价词语（中文）.txt')
neg_words = init_words('./doc/Hownet/负面情感词语（中文）.txt')
neg_words += init_words('./doc/Hownet/负面评价词语（中文）.txt')

print()
print(len(pos_words), len(neg_words), len(how_words))

pos_words = list(set(pos_words))
neg_words = list(set(neg_words))
how_words = list(set(how_words))
print(len(pos_words), len(neg_words), len(how_words))


def sentiment_words_count(cut_words):
    if len(cut_words) == 0:
        return [0, 0, 0]

    sentiment = []

    for words in [pos_words, neg_words]:
        c = 0
        for word in words:
            if word in cut_words:
                # print(word)
                c += 1
        sentiment.append(c)

    sentiment = [c / len(cut_words) for c in sentiment]

    c = 0
    for word in how_words:
        if word in cut_words:
            # print(word)
            c += how_words_dict[word]

    sentiment.append(c)
    return sentiment


# ============================== 大连理工词典 ==============================

_, words2array = joblib.load('./doc/大连理工大学情感词汇本体库/words2array_27351.pkl')
len(words2array), words2array['快乐'].shape


def get_not_and_how_value(cut_words, i, windows=2):
    not_cnt = 0
    how_v = 1

    left = 0 if (i - windows) < 0 else (i - windows)
    for w in cut_words[left:i]:
        if w in negation_words:
            not_cnt += 1
        if w in how_words:
            how_v *= how_words_dict[w]

    return (-1) ** not_cnt, how_v


def dalianligong_arr(cut_words, windows=2):
    arr = np.zeros(29)
    terms = list(words2array.keys())

    for i, word in enumerate(cut_words):
        if word in terms:
            not_v, how_v = get_not_and_how_value(cut_words, i, windows)
            arr += not_v * how_v * words2array[word]

            #             print('{}, {}, {}'.format(word, not_v, how_v))

    return arr


# ============================== BosonNLP 情感词典 ==============================


boson_words_dict = dict()
with open('./doc/BosonNLP情感词典/BosonNLP_sentiment_score.txt', 'r') as src:
    lines = src.readlines()
    for line in lines:
        boson_word = line.strip().split()
        if len(boson_word) != 2:
            print(line)
        else:
            boson_words_dict[boson_word[0]] = float(boson_word[1])

len(boson_words_dict)


def boson_value(cut_words, windows=2):
    value = 0

    for i, word in enumerate(cut_words):
        if word in boson_words_dict.keys():
            not_v, how_v = get_not_and_how_value(cut_words, i, windows)
            value += not_v * how_v * boson_words_dict[word]

            # print('{}, {}, {}, {}'.format(word, not_v, how_v, boson_words_dict[word]))

    return value


# ============================== Emoticon ==============================

emoticon_df = pd.read_csv('./doc/微博emoticon五分类标注/label_final.csv')
len(emoticon_df)

emoticon_df.head()
emoticon_df['label'].value_counts()

emoticon_types = list(set(emoticon_df['label'].tolist()))
emoticon_types.sort()
print(emoticon_types)

emoticon2index = dict(zip(emoticon_types, [i for i in range(len(emoticon_types))]))
emoticon2index

emoticons = emoticon_df['emoticon'].tolist()
len(emoticons), len(set(emoticons))


def emoticon_arr(text, cut_words):
    arr = np.zeros(len(emoticon_types))

    if len(cut_words) == 0:
        return arr

    labels = emoticon_df['label'].tolist()
    for i, emoticon in enumerate(emoticons):
        if emoticon in text:
            arr[emoticon2index[labels[i]]] += text.count(emoticon)

    return arr / len(cut_words)


# ============================== Punctuation ==============================

def symbols_count(text):
    excl = (text.count('!') + text.count('！')) / len(text)
    ques = (text.count('?') + text.count('？')) / len(text)
    comma = (text.count(',') + text.count('，')) / len(text)
    dot = (text.count('.') + text.count('。')) / len(text)
    ellip = (text.count('..') + text.count('。。')) / len(text)

    return excl, ques, comma, dot, ellip


# ============================== Others ==============================

first_pronoun = init_words('./doc/wiki/1-personal-pronoun.txt')
second_pronoun = init_words('./doc/wiki/2-personal-pronoun.txt')
third_pronoun = init_words('./doc/wiki/3-personal-pronoun.txt')
pronoun_words = [first_pronoun, second_pronoun, third_pronoun]


def pronoun_count(cut_words):
    if len(cut_words) == 0:
        return [0, 0, 0]

    pronoun = []
    for words in pronoun_words:
        c = 0
        for word in words:
            c += cut_words.count(word)
        pronoun.append(c)
    return [c / len(cut_words) for c in pronoun]


def negation_words_count(cut_words):
    if len(cut_words) == 0:
        return 0

    c = 0
    for word in negation_words:
        c += cut_words.count(word)

    return c / len(cut_words)


# ============================== Baidu API ==============================

baidu_emotions = ['angry', 'disgusting', 'fearful', 'happy', 'sad', 'neutral', 'pessimistic', 'optimistic']
baidu_emotions.sort()
baidu_emotions

baidu_emotions_2_index = dict(zip(baidu_emotions, [i for i in range(len(baidu_emotions))]))
baidu_emotions_2_index


def baidu_arr(emotions_dict):
    arr = np.zeros(len(baidu_emotions))

    for k, v in emotions_dict.items():
        # like -> happy
        if k == 'like':
            arr[baidu_emotions_2_index['happy']] += v
        else:
            arr[baidu_emotions_2_index[k]] += v

    return arr
