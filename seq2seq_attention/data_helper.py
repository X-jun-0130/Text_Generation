import tensorflow as tf
import random
import numpy as np
from collections import Counter

def Tokenizer(text, wordlist):
    text2id = []
    for word in text:
        if word in wordlist:
            text2id.append(wordlist.index(word))
        else:
            word = '[UNK]'
            text2id.append(wordlist.index(word)) 
    return text2id

def get_data(file):
    all_data =[key.strip('\n') for key in  open(file, 'r', encoding='utf-8')]
    random.shuffle(all_data)
    content_list = []
    word_list = []
    for line in  all_data[:900000]:
        line = line.strip()
        sen = line.split('<SEP>')
        input = sen[0].strip().split(' ')
        text = sen[1].strip().split(' ')
        content_list.append((input, text))

        word_list.extend(input)
        word_list.extend(text)
    
    word_count = Counter(word_list).most_common(44996)
    vocab_words = []
    for pari in word_count:
        vocab_words.append(pari[0])
    vocab_words.insert(0, '[PAD]')
    vocab_words.insert(1, '[UNK]')
    vocab_words.insert(2, '[GO]')
    vocab_words.insert(3, '[EOS]')
    print('finished load data')
    return content_list, vocab_words

def process(sentence):
    sentence.insert(0, '[GO]')
    sentence.append('[EOS]')
    return sentence

def padding(batch,word_list):
    input = [Tokenizer(key[0],word_list) for key in batch]
    text = [Tokenizer(process(key[1]),word_list) for key in batch]

    k_len = max([len(key) for key in input])
    w_len = max([len(key) for key in text])

    input = tf.keras.preprocessing.sequence.pad_sequences(input, k_len, padding='post', truncating='post')
    text = tf.keras.preprocessing.sequence.pad_sequences(text, w_len, padding='post', truncating='post')   

    return input, text


def next_batch(data, batch_size, word_list):
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
        input, text = padding(batch_data, word_list)
        yield input, text


# content_list, vocab_words = get_data('./single_generative/train_data.txt')
# print(content_list[0])
# print(len(vocab_words))
# print(vocab_words[:10])
