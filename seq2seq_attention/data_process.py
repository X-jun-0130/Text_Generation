
import tensorflow as tf
import json
import random
import numpy as np


key_word = open('./data/key_word.txt', 'r', encoding='utf-8')
keyword_list = [key.strip('\n') for key in key_word]


word2id = open('./data/value_word.txt', 'r', encoding='utf-8')
word_list = [key.strip('\n') for key in word2id]


def Tokenizer(text, wordlist):
    text2id = []
    for word in text:
        if word in wordlist:
            text2id.append(wordlist.index(word))
        else:
            word = '[UNK]'
            text2id.append(wordlist.index(word)) 
    return text2id

def get_data(all_data):
    content_list = []
    for i, k in enumerate(all_data):
        key = k[0]
        value = k[1]
        text = k[2]
        content_list.append((key, value, text))
    return content_list

def process(sentence):
    sentence.insert(0, '[GO]')
    sentence.append('[EOS]')
    return sentence

def padding(batch):
    key = [Tokenizer(key[0],keyword_list) for key in batch]
    value = [Tokenizer(key[1],word_list) for key in batch]
    text = [Tokenizer(process(key[2]),word_list) for key in batch]

    
    k_len = max([len(key) for key in key])
    w_len = max([len(key) for key in text])


    key = tf.keras.preprocessing.sequence.pad_sequences(key, k_len, padding='post', truncating='post')
    value = tf.keras.preprocessing.sequence.pad_sequences(value, k_len, padding='post', truncating='post')
    text = tf.keras.preprocessing.sequence.pad_sequences(text, w_len, padding='post', truncating='post')   

    return key, value, text


def next_batch(data, batch_size):
    """
    生成batch数据集
    :param data: 输入
    :param batch_size: 批量的大小
    :return:
    """
    random.shuffle(data)
    batch_num = len(data) // batch_size

    for i in range(batch_num):
        batch_data = data[batch_size * i: batch_size * (i + 1)]
        key, value, text = padding(batch_data)
        yield key, value, text
