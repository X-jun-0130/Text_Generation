import tensorflow as tf
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

embedding_dim = 256

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
    def __init__(self, units,vocab_size, keep_prob):
        super(Gene, self).__init__()
        self.units = units
        self.keep_peob = keep_prob
        self.vocab_size = vocab_size
        self.embedding = tf.keras.layers.Embedding(self.vocab_size,embedding_dim)
        
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
        self.luoatt = LuongAttention('dot', self.units)

    def encoder(self, inputs):
        value_embedding = self.embedding(inputs)
        output, state = self.gru(value_embedding, initial_state=tf.zeros((inputs.shape[0], self.units)))
        return output, state

    def decoder(self, x,  enc_output, source,  hidden):
        x = self.embedding(x)
        output, state = self.gru_decoder(x, initial_state=hidden)
        context_vector, _ = self.luoatt(state, enc_output, source)
        output = tf.concat((context_vector, output[:,-1,:]), axis=-1)
        output = self.dropout(output)
        # 输出的形状 == （批大小，vocab）
        output = self.dense(output)
        return output, state


def loss_function(real, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)
