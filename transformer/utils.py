import tensorflow as tf
import os
import re
import json

CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
BLEU_CALCULATOR_PATH = os.path.join(CURRENT_DIR_PATH, 'multi-bleu.perl')

class Mask:
    @classmethod
    def create_padding_mask(cls, sequences):
        sequences = tf.cast(tf.math.equal(sequences, 0), dtype=tf.float32)
        return sequences[:, tf.newaxis, tf.newaxis, :]

    @classmethod
    def create_look_ahead_mask(cls, seq_len):
        return 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

    @classmethod
    def create_masks(cls, inputs, target):
        encoder_padding_mask = Mask.create_padding_mask(inputs)
        decoder_padding_mask = Mask.create_padding_mask(inputs)

        look_ahead_mask = tf.maximum(
            Mask.create_look_ahead_mask(tf.shape(target)[1]),
            Mask.create_padding_mask(target)
            )

        return encoder_padding_mask, look_ahead_mask, decoder_padding_mask


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def label_smoothing(target_data, depth, epsilon=0.1):
    target_data_one_hot = tf.one_hot(target_data, depth=depth)
    n = target_data_one_hot.get_shape().as_list()[-1]
    return ((1 - epsilon) * target_data_one_hot) + (epsilon / n)


def loss_function(real, pred, vocab_size):
    loss_object = tf.losses.CategoricalCrossentropy(from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    real_one_hot = label_smoothing(real, depth=vocab_size)
    loss = loss_object(real_one_hot, pred)

    mask = tf.cast(mask, dtype=loss.dtype)

    loss *= mask
    return tf.reduce_mean(loss)

def translate(inputs,  model, seq_max_len_target):
    _str = ''
    with open(os.path.join('./data/', "word_to_index.json"), "r", encoding="utf8") as f:
            word_to_idx = json.load(f)
    
    idx_to_label = {value: key for key, value in word_to_idx.items()}

    word_idx = [word_to_idx.get(token, word_to_idx["<UNK>"]) for token in inputs.split(' ')]

    encoder_inputs = tf.convert_to_tensor(
        [word_idx],
        dtype=tf.int32
    )
    decoder_inputs = [word_to_idx['<GO>']]
    decoder_inputs = tf.expand_dims(decoder_inputs, 0)
    decoder_end_token = word_to_idx['<EOS>']

    for _ in range(seq_max_len_target):
        encoder_padding_mask, look_ahead_mask, decoder_padding_mask = Mask.create_masks(
            encoder_inputs, decoder_inputs
        )
        pred = model(
            inputs=encoder_inputs,
            target=decoder_inputs,
            inputs_padding_mask=encoder_padding_mask,
            look_ahead_mask=look_ahead_mask,
            target_padding_mask=decoder_padding_mask,
            training=False
        )
        pred = pred[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(pred, axis=-1), dtype=tf.int32)

        if predicted_id[0] == decoder_end_token:
            break

        decoder_inputs = tf.concat([decoder_inputs, predicted_id], axis=-1)

    total_output = tf.squeeze(decoder_inputs, axis=0)
    _str = ' '.join([idx_to_label[int(key)] for key in total_output])
    return _str


def calculate_bleu_score(target_path, ref_path):

    get_bleu_score = f"perl {BLEU_CALCULATOR_PATH} {ref_path} < {target_path} > temp"
    os.system(get_bleu_score)
    with open("temp", "r") as f:
        bleu_score_report = f.read()
    score = re.findall("BLEU = ([^,]+)", bleu_score_report)[0]

    return score, bleu_score_report


