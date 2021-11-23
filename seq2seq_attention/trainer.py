import json
from model import Gene, loss_function
import tensorflow as tf
from data_helper import get_data, next_batch, Tokenizer
import numpy as np
import random
import os

lr = 0.0005
num_epochs = 20
BATCH_SIZE = 32
clip = 5.0
embedding_dim = 256
units = 512
keep_prob = 0.7

content_list, vocab_words = get_data('./single_generative/train_data.txt')
vocab_size = len(vocab_words)

G_model = Gene(units, vocab_size, keep_prob)

def train_step(inp,  targ):
    loss = 0
    with tf.GradientTape() as tape:
        source = inp
        enc_output, enc_hidden = G_model.encoder(inp)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([2] * BATCH_SIZE, 1)
        output_probs = tf.zeros([BATCH_SIZE, 1, vocab_size])
        # 教师强制 - 将目标词作为下一个输入
        for t in range(1, targ.shape[1]):
            # 将编码器输出 （enc_output） 传送至解码器
            predictions,  dec_hidden = G_model.decoder(dec_input,  enc_output, source, dec_hidden)
            output_probs = tf.concat((output_probs,tf.expand_dims(predictions, 1)), 1)
            # 使用教师强制，将之前输出全部作为输入
            dec_input = targ[:, 0:t+1]
        
        loss = loss_function(targ[:,1:], output_probs[:,1:,:])
    variables = G_model.variables
    gradients = tape.gradient(loss, variables)
    gradients, _ = tf.clip_by_global_norm(gradients, clip)
    optimizer.apply_gradients(grads_and_vars=zip(gradients, variables))

    return loss

def evaluate(sentence):
    sentence = sentence.strip().split(' ')
    init_state = [2]
    dec_input = tf.expand_dims(init_state, 0)

    value = [Tokenizer(sentence,vocab_words)]
    in_value = tf.convert_to_tensor(value)

    result = ''

    enc_out, enc_hidden = G_model.encoder(in_value)
    dec_hidden = enc_hidden
    source = in_value

    for _ in range(65):
        predictions, dec_hidden = G_model.decoder(dec_input, enc_out, source, dec_hidden)
        predicted_id = tf.argmax(predictions[0]).numpy()
        if vocab_words[predicted_id] == '[EOS]':
            break
        else:
            result += vocab_words[predicted_id] + ' '
            init_state.append(predicted_id)
    
            # 预测的 ID 被输送回模型
            dec_input = tf.expand_dims(init_state, 0)
    
    return result

random.shuffle(content_list)
train = content_list

pre_sentence = '孩纸 。 你 绝对 是 未来 的 杰克逊 哦 ！'

batch_index = 1
best = 1.8
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
checkpoint = tf.train.Checkpoint(optimizer=optimizer,model=G_model)
for epoch in range(num_epochs):
    print('Epoch:', epoch + 1)
    batch_train = next_batch(train, BATCH_SIZE, vocab_words)
    for input, text in batch_train:
        batch_index += 1
        batch_loss = train_step(input, text)

        if batch_index % 1000 == 0:
            print('Batch {} Loss {:.4f}'.format(batch_index, batch_loss))

    if best > batch_loss:
        best = batch_loss
        print(evaluate(pre_sentence))
        print('saving_model')
        checkpoint.save('./seq2seq/save/generative_model.ckpt')


