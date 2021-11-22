import json
from phvm_model import Gene, loss_function
import tensorflow as tf
from phvm_data_help import get_data, next_batch, keyword_list,  process
import numpy as np
import random
import os

lr = 0.0005
num_epochs = 40
key_dim = 256
value_dim = 256
BATCH_SIZE = 16
clip = 5.0

embedding_dim = 256
units = 512
keep_prob = 0.7
key_word_size = len(keyword_list)

vocab_dict, reverse_vocab_dict , vocab_size = process()
G_model = Gene(units, vocab_size, key_word_size, keep_prob)

def accurucy(output_probs, decoder_outputs):
    decoder_seq_len = tf.reduce_sum(tf.sign(decoder_inputs), axis=1)
    decoder_max_len = tf.shape(decoder_inputs)[1]
    decoder_mask = tf.sequence_mask(decoder_seq_len, decoder_max_len)
    nonzeros = tf.math.count_nonzero(decoder_mask)
    decoder_mask = tf.cast(decoder_mask, dtype=tf.float32)

    predict = tf.argmax(output_probs, 2)
    correct_pred = tf.equal(tf.cast(predict, tf.int32),decoder_outputs)
    accuracy = tf.divide(tf.reduce_sum(tf.cast(correct_pred, tf.float32) * decoder_mask),tf.cast(nonzeros, tf.float32))
    each_accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.float32) * decoder_mask, 1) / tf.reduce_sum(decoder_mask, 1)
    return accuracy

def train_step(inp, inp_value, targ, decoder_outputs, batch_OOV_num):
    output_probs= []
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = G_model.encoder(inp, inp_value)
        dec_hidden = enc_hidden
        output_probs = tf.zeros([BATCH_SIZE, 1, vocab_size+ batch_OOV_num])
        # 教师强制 - 将目标词作为下一个输入
        for t in range(0, targ.shape[1]):
            dec_input = tf.expand_dims(targ[:, t], 1)
            # 将编码器输出 （enc_output） 传送至解码器
            predictions,  dec_hidden = G_model.decoder(dec_input, inp_value, enc_output, dec_hidden, batch_OOV_num)
            output_probs = tf.concat((output_probs,tf.expand_dims(predictions, 1)), 1)
            # 使用教师强制，将t-1输出作为输入
        output_probs = output_probs[:,1:,:]
        loss = loss_function(decoder_inputs, decoder_outputs, vocab_size, batch_OOV_num, output_probs)

    accu = accurucy(output_probs, decoder_outputs)
    #batch_loss = (loss / int(targ.shape[1]))
    variables = G_model.trainable_variables
    gradients = tape.gradient(loss, variables)
    gradients, _ = tf.clip_by_global_norm(gradients, clip)
    optimizer.apply_gradients(grads_and_vars=zip(gradients, variables))

    return loss, accu 

def evaluate(key, sentence):
    
    source_key, source_sent= [],[]

    source_key.append(key)
    source_sent.append(sentence)

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

    result = ''
    enc_out, enc_hidden = G_model.encoder(encoder_key, encoder_inputs)
    dec_hidden = enc_hidden
    state = 1
    dec_input = tf.expand_dims([vocab_dict['<BOS>']], 0)
    while state:
        predictions, dec_hidden = G_model.decoder(dec_input, encoder_inputs, enc_out, dec_hidden, batch_OOV_num)
        predicted_id = tf.argmax(predictions[0]).numpy()

        if predicted_id < vocab_size:
            result += reverse_vocab_dict[predicted_id] + ' '
        else:
            result += batch_OOV_tokens[0][int(predicted_id) - vocab_size] + ' '
        
        # 预测的 ID 被输送回模型
        dec_input = tf.expand_dims([predicted_id], 0)
        
        if '<EOS>' in result:
            break

    return result



data= json.load(open('./data/phvm_data.json', 'r', encoding='utf-8'))
content = get_data(data)
random.shuffle(content)
#content = content[:45000]
n =  int(len(content) * 0.99)
train = content[:n]
test = content[n:]


kk =     [
      "类型",
      "材质",
      "风格",
      "图案",
      "衣样式",
      "衣袖型",
      "衣款式",
      "衣款式",
      "衣款式"
    ]

kk_value =      [
      "上衣",
      "丝绒",
      "复古",
      "复古",
      "雪纺衫",
      "喇叭袖",
      "木耳边",
      "飘带",
      "荷叶边"
    ]
batch_index = 1
best = 2.5
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
checkpoint = tf.train.Checkpoint(optimizer=optimizer,model=G_model)
for epoch in range(num_epochs):
    print('Epoch:', epoch + 1)
    batch_train = next_batch(train, BATCH_SIZE,vocab_dict, vocab_size)
    for encoder_key,encoder_inputs, decoder_inputs, decoder_outputs, batch_OOV_tokens, batch_OOV_num in batch_train:
        batch_index += 1
        batch_loss, accu = train_step(encoder_key,encoder_inputs, decoder_inputs, decoder_outputs, batch_OOV_num)
        if batch_index % 500 == 0:
            print('Batch {} Loss {:.4f}  accu {:.4f}'.format(batch_index, batch_loss, accu))

    if best > batch_loss:
        best = batch_loss
        print(evaluate(kk, kk_value))
        print('saving_model')
        checkpoint.save('./seq2seq/phvm_save/generative_model.ckpt')
