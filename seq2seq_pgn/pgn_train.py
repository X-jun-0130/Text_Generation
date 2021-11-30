import json
from PGN_model import PGN, _calc_final_dist
import tensorflow as tf
from data_helper import get_data, next_batch, keyword_list,  process
import numpy as np
import random
from loss import calc_loss

lr = 0.0005
num_epochs = 100
BATCH_SIZE = 16
clip = 5.0

embedding_dim = 256
units = 256
key_word_size = len(keyword_list)

vocab_dict, reverse_vocab_dict , vocab_size = process()
G_model = PGN(units)

random_p = [0, 1, 1, 1, 1, 1, 1, 1,1,1]

def train_step(inp, inp_value, dec_inp,dec_outputs, enc_extended_inp, batch_oov_len, enc_mask, dec_mask, cov_loss_wt):
    with tf.GradientTape() as tape:
        final_dist, attentions, coverages = G_model(inp, inp_value, dec_inp, enc_extended_inp, batch_oov_len, enc_mask, True)
        batch_loss, log_loss, cov_loss = calc_loss(dec_outputs, final_dist, dec_mask, attentions, coverages, cov_loss_wt, True)

    variables = G_model.trainable_variables
    gradients = tape.gradient(batch_loss, variables)
    gradients, _ = tf.clip_by_global_norm(gradients, clip)
    optimizer.apply_gradients(grads_and_vars=zip(gradients, variables))
    return batch_loss, log_loss, cov_loss 

def evaluate(key, sentence):
    source_key, source_sent= [],[]
    source_key.append(key)
    source_sent.append(sentence)
    batchsize = len(source_sent)

    encoder_max_length = 0
    for sent in source_sent:
        encoder_max_length = max(encoder_max_length, len(sent))
    encoder_inputs = np.zeros(dtype=int, shape=(batchsize, encoder_max_length))
    encoder_extend_vocab = np.zeros(dtype=int, shape=(batchsize, encoder_max_length)) 
    encoder_key = np.zeros(dtype=int, shape=(batchsize, encoder_max_length))
    enc_mask = np.zeros(dtype=float, shape=(batchsize, encoder_max_length))

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
            enc_mask[i][j] = 1.0
            if token in vocab_dict:
                encoder_extend_vocab[i][j] = vocab_dict[token]
                encoder_inputs[i][j] = vocab_dict[token]
            else:
                encoder_inputs[i][j] = vocab_dict['<UNK>']
                encoder_extend_vocab[i][j] = vocab_size + OOV_token.index(token)
            
    batch_OOV_num = np.max([len(tokens) for tokens in batch_OOV_tokens])

    result = ''
    enc_out, enc_hidden = G_model.call_encoder(encoder_key, encoder_inputs)
    dec_hidden = enc_hidden
    state = 1
    dec_input = tf.expand_dims(vocab_dict['<BOS>'], 0)
    coverage = tf.zeros((enc_out.shape[0], enc_out.shape[1], 1))
    while state < 135:
        final_dist, dec_hidden, context_vector, attn, p_gen, coverage = G_model.call_decoder_one_step(encoder_extend_vocab, batch_OOV_num, dec_input, dec_hidden, enc_out, enc_mask, coverage, True)
        final_dist = tf.squeeze(final_dist, axis=1)
        predicted_id = tf.argmax(final_dist[0]).numpy()
        if predicted_id < vocab_size:
            result += reverse_vocab_dict[predicted_id] + ' '
        else:
            result += batch_OOV_tokens[0][int(predicted_id) - vocab_size] + ' '
        
        # 预测的 ID 被输送回模型
        dec_input = tf.argmax(final_dist, axis=1)

        state += 1
        if '<EOS>' in result:
            break

    return result


data= json.load(open('./long_data/phvm_data.json', 'r', encoding='utf-8'))
content = get_data(data)
random.shuffle(content)
#content = content[:45000]
n =  int(len(content) * 0.99)
train = content
# test = content[n:]

kk =     [
      "类型",
      "材质",
      "颜色",
      "图案",
      "图案",
      "裙腰型",
      "裙长",
      "裙袖长",
      "裙领型"
    ]

kk_value =      [
      "裙",
      "网纱",
      "粉红色",
      "线条",
      "刺绣",
      "高腰",
      "连衣裙",
      "短袖",
      "圆领"
    ]

batch_index = 1
best = 3.0
cov_loss_wt = 0.5
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
#optimizer = tf.keras.optimizers.Adagrad(lr,initial_accumulator_value= 0.1,clipnorm=2.0,epsilon=1e-12)
checkpoint = tf.train.Checkpoint(optimizer=optimizer,model=G_model)
for epoch in range(num_epochs):
    print('Epoch:', epoch + 1)
    batch_train = next_batch(train, BATCH_SIZE,vocab_dict, vocab_size)
    for encoder_key,encoder_inputs, encoder_extend_vocab, decoder_inputs, decoder_outputs, batch_OOV_tokens, batch_OOV_num, enc_mask, dec_mask in batch_train:
        batch_index += 1
        batch_loss, log_loss, cov_loss = train_step(encoder_key,encoder_inputs, decoder_inputs, decoder_outputs, encoder_extend_vocab, batch_OOV_num, enc_mask, dec_mask, cov_loss_wt)
        if batch_index % 300 == 0:
            print('Batch {} Loss {:.4f}  cov_loss {:.4f}'.format(batch_index, batch_loss, cov_loss))

    if best > batch_loss:
        best = batch_loss
        print(evaluate(kk, kk_value))
        print('saving_model')
        checkpoint.save('./PGN_generative/save/generative_model.ckpt')