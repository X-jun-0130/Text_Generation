
import json
import random
import numpy as np
import copy
from collections import Counter


import tensorflow as tf#
key_word = open('./data/phvm_key_word.txt', 'r', encoding='utf-8')
keyword_list = [key.strip('\n') for key in key_word]


word2id = open('./data/phvm_value_word.txt', 'r', encoding='utf-8')
word_list = [key.strip('\n') for key in word2id]


def get_data(all_data):
    content_list = []
    for i, k in enumerate(all_data):
        key = k[0]
        value = k[1]
        text = k[2]
        content_list.append((key, value, text))
    return content_list

def process():
    all_tokens = {}
    all_tokens['<PAD>'] = 0
    for ch in word_list:
        all_tokens[ch] = len(all_tokens)
    all_tokens['<BOS>'] = len(all_tokens)
    all_tokens['<EOS>'] = len(all_tokens)
    all_tokens['<UNK>'] = len(all_tokens)
    vocab_size = len(all_tokens)
    # print(vocab_size)
    # treat 连衣裙,文字,撞色... as OOV
    vocab_dict = copy.deepcopy(all_tokens)
    '''
    连衣裙
    文字
    撞色
    朋克
    复古
    '''
    reverse_vocab_dict = dict(zip(vocab_dict.values(), vocab_dict.keys()))
    return vocab_dict, reverse_vocab_dict , vocab_size


def tok2idx(vocab_dict, vocab_size, batch_data):
    source_key, source_sent, target_sent = [],[],[]
    for key in batch_data:
        source_key.append(key[0])
        source_sent.append(key[1])
        target_sent.append(key[2])

    batchsize = len(source_sent)

    encoder_max_length = 0
    for sent in source_sent:
        encoder_max_length = max(encoder_max_length, len(sent))
    encoder_inputs = np.zeros(dtype=int, shape=(batchsize, encoder_max_length))
    encoder_key = np.zeros(dtype=int, shape=(batchsize, encoder_max_length))

    batch_OOV_tokens = []

    for x, key in enumerate(source_key):
        for k, k_w in enumerate(key):
            if k_w in keyword_list:
                encoder_key[x][k] = keyword_list.index(k_w)
            else:
                encoder_inputs[x][k] = 1

    for i, sent in enumerate(source_sent):
        OOV_token = list(set([w for w in sent if w not in vocab_dict]))
        batch_OOV_tokens.append(OOV_token)
        for j, token in enumerate(sent):
            if token in vocab_dict:
                encoder_inputs[i][j] = vocab_dict[token]
            else:
                encoder_inputs[i][j] = vocab_size + OOV_token.index(token)
    
    batch_OOV_num = np.max([len(tokens) for tokens in batch_OOV_tokens])

    decoder_max_length = 0
    for sent in target_sent:
        decoder_max_length = max(decoder_max_length, len(sent))

    decoder_inputs = np.zeros(dtype=int, shape=(batchsize, decoder_max_length + 1))
    decoder_outputs = np.zeros(dtype=int, shape=(batchsize, decoder_max_length + 1))
    for i, sent in enumerate(target_sent):
        for j, token in enumerate(sent):
            if token in vocab_dict:
                decoder_inputs[i][j + 1] = vocab_dict[token]
                decoder_outputs[i][j] = vocab_dict[token]

            elif token in batch_OOV_tokens[i]:
                decoder_inputs[i][j + 1] = vocab_size + batch_OOV_tokens[i].index(token)
                decoder_outputs[i][j] = vocab_size + batch_OOV_tokens[i].index(token)

            else:
                decoder_inputs[i][j + 1] = vocab_dict['<UNK>']
                decoder_outputs[i][j] = vocab_dict['<UNK>']

        decoder_inputs[i][0] = vocab_dict['<BOS>']
        decoder_outputs[i][len(sent)] = vocab_dict['<EOS>']


    return encoder_key,encoder_inputs, decoder_inputs, decoder_outputs, batch_OOV_tokens,batch_OOV_num

def next_batch(data, batch_size, vocab_dict, vocab_size):
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
        encoder_key,encoder_inputs, decoder_inputs, decoder_outputs, batch_OOV_tokens,batch_OOV_num = tok2idx(vocab_dict, vocab_size, batch_data)
        yield encoder_key,encoder_inputs, decoder_inputs, decoder_outputs, batch_OOV_tokens, batch_OOV_num
#[('上衣', 16285), ('裙', 16150), ('裤', 8875)]
# all_data = []
# def get_text(filename):
#     key_word_list = []
#     word_list = []
#     max_length = []
#     ta =  open(filename, 'rb')
#     for line in ta:
#         line = json.loads(line)
#         fea = line['feature']
#         desc = [key for key in line['desc'].split(' ') if key != '<UNK>']
#         if len(desc) <= 60:
#             key = [key[0] for key in fea]
#             value = [key[1] for key in fea]
#             key_word_list.extend(key)
#             max_length.append(value[0])
#             word_list.extend(value)
#             word_list.extend(desc)
#             all_data.append((key, value, desc))
#         else:
#             continue

#     word_type = Counter(max_length).most_common(5)
#     return all_data, key_word_list, word_list, word_type

# ta, word_ta, word_list , max_length = get_text('./data/train.jsonl')
# print(len(ta))
# print(max_length)

# train_data = open('./data/phvm_data.json', 'wb')
# eachline = json.dumps(all_data, ensure_ascii=False, indent=2) + '\n'
# eachline = eachline.encode()
# train_data.write(eachline)


# key_word = open('./data/phvm_key_word.txt', 'w', encoding='utf-8')
# for key in list(set(word_ta)):
#     key_word.write(key+'\n')

# _word = open('./data/phvm_value_word.txt', 'w', encoding='utf-8')
# for kk in list(set(word_list)):
#     _word.write(kk+'\n')

