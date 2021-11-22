import tensorflow as tf
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

key_dim = 256
value_dim = 256
embedding_dim = 256
attention_size = 256

key_word = open('./data/phvm_key_word.txt', 'r', encoding='utf-8')
keyword_list = [key.strip('\n') for key in key_word]

word2id = open('./data/phvm_value_word.txt', 'r', encoding='utf-8')
word_list = [key.strip('\n') for key in word2id]

def att(output):
    u_list = []
    seq_size = output.shape[1]
    hidden_size = output.shape[2] #[2*hidden_dim]
    attention_w = tf.random_normal_initializer(mean=0.0, stddev=0.1)([hidden_size, attention_size])
    attention_u = tf.random_normal_initializer(mean=0.0, stddev=0.1)([attention_size, 1])
    attention_b = tf.Variable(tf.constant(0.1, shape=[attention_size]), name='attention_b')
    for t in range(seq_size):
        #u_t:[1,attention]
        u_t = tf.tanh(tf.matmul(output[:, t, :], attention_w) + tf.reshape(attention_b, [1, -1]))
        u = tf.matmul(u_t, attention_u)
        u_list.append(u)
    logit = tf.concat(u_list, axis=1)
    #u[seq_size:attention_z]
    weights = tf.nn.softmax(logit, name='attention_weights')
    out_final = tf.reduce_sum(output * tf.reshape(weights, [-1, seq_size, 1]), 1)
    return out_final 


class LuongAttention(tf.keras.layers.Layer):
    '''
    Reference:
     "Effective approaches to Attention-based Neural Machine Translation"
     https://arxiv.org/abs/1508.04025
    '''
    def __init__(self, method, attn_unit, **kwargs):
        super(LuongAttention, self).__init__()
        
        # method: indicator that determines the score function
        self.attn_unit = attn_unit
        self.method = method

        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention score function.")

    def dot_score(self, query, values):
        # [before reduce_sum] shape == (batch_size, enc_max_len, enc_unit)
        # [after reduce_sum] shape == (batch_size, enc_max_len)
        return tf.reduce_sum(query * values, axis=2)
        
    def call(self, query, values, source=None):
        # query: decoder hidden state at current time step t
        # values: encoder outputs
        # source: encoder inputs which are represented with indexes (for calculating masked softmax)
        # query shape == (batch_size, dec_unit)
        # values shape == (batch_size, enc_max_len, enc_unit)
        # source shape == (batch_size, enc_max_len)
        
        # query_hidden shape == (batch_size, 1, dec_unit)
        # score shape == (batch_size, enc_max_len, 1)
        query_hidden = tf.expand_dims(query, 1)

        if self.method == 'dot':
            attn_score = self.dot_score(query_hidden, values)
        
        # masking attention score corresponding to pad values in encoder inputs
        # attention_weights shape == (batch_size, enc_max_len, 1)
        masked_score = self.masking_attn_score(source, attn_score)
        masked_score = masked_score[:, :, tf.newaxis]
        attention_weights = tf.nn.softmax(masked_score, axis=1)

        # context_vector (before sum) shape == (batch_size, enc_max_len, enc_unit)
        # context_vector (after sum) shape == (batch_size, enc_unit)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights
        
    def masking_attn_score(self, source, score):
        # The mask is multiplied with -1e9 (close to negative infinity)
        # The large negative inputs to softmax are near zero in the output
        # because the output of softmax function is normalized into 0~1
        # source shape == (batch_size, enc_max_len)
        # score shape == (batch_size, enc_max_len)
        mask = tf.math.logical_not(tf.math.not_equal(source, 0))
        mask = tf.cast(mask, dtype=score.dtype)
        mask *= -1e9

        return score + mask
        

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # 隐藏层的形状 == （批大小，隐藏层大小）
        # hidden_with_time_axis 的形状 == （批大小，1，隐藏层大小）
        # 这样做是为了执行加法以计算分数
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # 分数的形状 == （批大小，最大长度，1）
        # 最后一个轴上得到 1， 把分数应用于 self.V
        # 在应用 self.V 之前，张量的形状是（批大小，最大长度，单位）
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        # 注意力权重 （attention_weights） 的形状 == （批大小，最大长度，1）
        attention_weights = tf.nn.softmax(score, axis=1)

        # 上下文向量 （context_vector） 求和之后的形状 == （批大小，隐藏层大小）
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Gene(tf.keras.Model):
    def __init__(self, units, vocab_size, key_word_size, keep_prob):
        super(Gene, self).__init__()
        self.units = units
        self.vocab_size = vocab_size
        self.key_size = key_word_size
        self.keep_peob = keep_prob
        self.key_embedding = tf.keras.layers.Embedding(self.key_size, embedding_dim)
        self.random_normal = tf.random_normal_initializer()
        self.w_emd = tf.Variable(self.random_normal(shape=[self.vocab_size, embedding_dim]), dtype=tf.float32, trainable=True)
    
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

        self.gru_decoder = tf.keras.layers.GRU(self.units,
                                return_sequences=True,
                                return_state=True,
                                recurrent_initializer='glorot_uniform')

        self.dropout = tf.keras.layers.Dropout(self.keep_peob)
        self.dense = tf.keras.layers.Dense(self.vocab_size)
        self.dense_hidden = tf.keras.layers.Dense(self.units)
        self.bahatt = BahdanauAttention(self.units)
    def mask_logits(self, seq_mask, scores):
        '''
        to do softmax, assign -inf value for the logits of padding tokens
        '''
        score_mask_values = -1e10 * tf.ones_like(scores, dtype=tf.float32)
        return tf.where(seq_mask, scores, score_mask_values)

    def encoder(self, key_inputs, value_inputs):
        encoder_key_emd = self.key_embedding(key_inputs)
        encoder_input_emb = tf.nn.embedding_lookup(params=self.w_emd,
        ids=tf.where(condition=tf.less(value_inputs, self.vocab_size),
                        x=value_inputs,
                        y=tf.ones_like(value_inputs) * self.vocab_size-1))

        x = tf.concat((encoder_key_emd, encoder_input_emb), 2)
        output, state = self.gru(x, initial_state=tf.zeros((key_inputs.shape[0], self.units)))
        return  output, state

    def decoder(self, x, enc_inputs,  enc_output, hidden, batch_OOV_num):
        decoder_input_emb = tf.nn.embedding_lookup(params=self.w_emd,
        ids=tf.where(condition=tf.less(x, self.vocab_size),
                        x=x,
                        y=tf.ones_like(x) * self.vocab_size-1))

        context_vector, _ = self.bahatt(hidden, enc_output)
        selective_mask = tf.cast(tf.equal(enc_inputs, x),dtype=tf.float32)  # batch * encoder_max_len
        # selective_mask_sum = tf.reduce_sum(selective_mask, axis=1)
        # select_less = tf.less(selective_mask_sum, 1e-10)
        # y = selective_mask / (tf.expand_dims(selective_mask_sum, 1) + 10**-10)
        #rou = tf.where(select_less, selective_mask, y)
        rou = selective_mask

        selective_read = tf.einsum("ijk,ij->ik", enc_output, rou)

        output = tf.concat((tf.expand_dims(context_vector, 1), tf.expand_dims(selective_read,1), decoder_input_emb), axis=-1)
        output, state = self.gru_decoder(output, initial_state=hidden)
        output = tf.reshape(output, shape=[-1, output.shape[-1]])
        # 输出的形状 == （批大小，1, hidden）
        '''
        generate_mechanism
        '''
        generate_score =  self.dense(output)  # batch * vocab_size
        '''
        copy_mechanism
        '''
        encoder_seq_len = tf.reduce_sum(tf.sign(enc_inputs), axis=1)
        encoder_max_len = tf.shape(enc_inputs)[1]
        encoder_mask = tf.sequence_mask(encoder_seq_len, encoder_max_len)
        #copy_score = tf.einsum("ijk,km->ijm", enc_output, self.weights_copy)
        copy_score  = self.dense_hidden(enc_output)
        copy_score = tf.nn.tanh(copy_score)
        copy_score = tf.einsum("ijm,im->ij", copy_score, output)
        copy_score = self.mask_logits(encoder_mask, copy_score)

        mix_score = tf.concat([generate_score, copy_score], axis=1)  # batch * (vocab_size + encoder_max_len)
        probs = tf.cast(tf.nn.softmax(mix_score), tf.float32)
        prob_g = probs[:, :self.vocab_size]
        prob_c = probs[:, self.vocab_size:]

        encoder_inputs_one_hot = tf.one_hot(indices=enc_inputs,depth=self.vocab_size + batch_OOV_num)
        prob_c = tf.einsum("ijn,ij->in", encoder_inputs_one_hot, prob_c)

        # if encoder inputs has intersection words with vocab dict,
        # move copy mode probability to generate mode probability

        prob_g = prob_g + prob_c[:, :self.vocab_size]
        prob_c = prob_c[:, self.vocab_size:]
        prob_final = tf.concat([prob_g, prob_c], axis=1) + 1e-10  # batch * (vocab_size + OOV_size)

        return prob_final, state


def loss_function(decoder_inputs, decoder_outputs, vocab_size, batch_OOV_num, output_probs):
    decoder_seq_len = tf.reduce_sum(tf.sign(decoder_inputs), axis=1)
    decoder_max_len = tf.shape(decoder_inputs)[1]
    decoder_mask = tf.sequence_mask(decoder_seq_len, decoder_max_len)
    decoder_outputs_one_hot = tf.one_hot(decoder_outputs, vocab_size + batch_OOV_num)
    crossent = - tf.reduce_sum(decoder_outputs_one_hot * tf.math.log(output_probs), -1)
    nonzeros = tf.math.count_nonzero(decoder_mask)
    decoder_mask = tf.cast(decoder_mask, dtype=tf.float32)
    train_loss = (tf.reduce_sum(crossent * decoder_mask) /
                        tf.cast(nonzeros, tf.float32))
    return train_loss
