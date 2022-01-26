from __future__ import (absolute_import, division, print_function,unicode_literals)

import os
import json
import tensorflow as tf
from seq2seq_model import Transformer
from utils import CustomSchedule,  Mask, CustomSchedule, loss_function, translate
from data_helper import TransformerSeq2SeqData
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# hyper paramaters
TRAIN_RATIO = 0.9
D_POINT_WISE_FF = 2048
D_MODEL = 512
ENCODER_COUNT = DECODER_COUNT = 6
EPOCHS = 100
ATTENTION_HEAD_COUNT = 8
DROPOUT_PROB = 0.1
BATCH_SIZE = 32
BPE_VOCAB_SIZE = 17501

DATA_LIMIT = None

GLOBAL_BATCH_SIZE = (BATCH_SIZE * 1)
print('GLOBAL_BATCH_SIZE ', GLOBAL_BATCH_SIZE)

with open('./config.json', "r", encoding='utf-8') as fr:
    config = json.load(fr)


data_obj = TransformerSeq2SeqData(config)

dataset = data_obj.gen_data(config["train_data"])
eval_data = data_obj.gen_data(config["eval_data"], is_training=False)

transformer = Transformer(
    vocab_size=BPE_VOCAB_SIZE,
    encoder_count=ENCODER_COUNT,
    decoder_count=DECODER_COUNT,
    attention_head_count=ATTENTION_HEAD_COUNT,
    d_model=D_MODEL,
    d_point_wise_ff=D_POINT_WISE_FF,
    dropout_prob=DROPOUT_PROB
)

learning_rate = CustomSchedule(D_MODEL)
optimizer = tf.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

senten = '裙 显瘦 蕾丝 蕾丝 抹胸裙 高腰 连衣裙 钉珠 吊带 收腰'

class Trainer:
    def __init__(
            self,
            model,
            dataset,
            optimizer=None,
            checkpoint_dir='./transformer/checkpoints/transformer.ckpt',
            batch_size=None,
            distribute_strategy=None,
            vocab_size=17501,
            epoch=30,
    ):
        self.batch_size = batch_size
        self.distribute_strategy = distribute_strategy
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        self.vocab_size = vocab_size
        self.epoch = epoch
        self.dataset = dataset

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)

        # metrics
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
        self.validation_loss = tf.keras.metrics.Mean('validation_loss', dtype=tf.float32)
        self.validation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('validation_accuracy')

    def multi_gpu_train(self):
        with self.distribute_strategy.scope():
            self.dataset = self.distribute_strategy.experimental_distribute_dataset(self.dataset)
            self.trainer(is_distributed=True)

    def single_gpu_train(self):
        self.trainer(is_distributed=False)
    

    def basic_train_step(self, inputs, target):
        target_include_start = target[:, :-1]
        target_include_end = target[:, 1:]
        encoder_padding_mask, look_ahead_mask, decoder_padding_mask = Mask.create_masks(inputs, target_include_start)

        with tf.GradientTape() as tape:
            pred = self.model.call(
                inputs=inputs,
                target=target_include_start,
                inputs_padding_mask=encoder_padding_mask,
                look_ahead_mask=look_ahead_mask,
                target_padding_mask=decoder_padding_mask,
                training=True
            )

            loss = loss_function(target_include_end, pred, self.vocab_size)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(target_include_end, pred)

        if self.distribute_strategy is None:
            return tf.reduce_mean(loss)

        return loss
    
    def train_step(self, inputs, target):
        return self.basic_train_step(inputs, target)

    def distributed_train_step(self, inputs, target):
        loss = self.distribute_strategy.experimental_run_v2(self.basic_train_step, args=(inputs, target))
        loss_value = self.distribute_strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)
        return tf.reduce_mean(loss_value)

    def trainer(self, is_distributed=False):
        batch = 0
        best = 0.15
        for epoch in range(self.epoch):
            ave_loss = []
            ave_accu = []
            print('start learning')

            for inputs, target in data_obj.next_batch(self.dataset, self.batch_size):
                batch += 1
                if is_distributed:
                    self.distributed_train_step(inputs, target)
                else:
                    self.train_step(inputs, target)

                ave_loss.append(self.train_loss.result())
                ave_accu.append(self.train_accuracy.result())
                if batch % 300 == 0:
                    print("Epoch: {}, Batch: {}, Loss:{}, Accuracy: {}".format(epoch+1, batch, self.train_loss.result(),self.train_accuracy.result()))
 
            print('Epoch:', epoch + 1)
            print('Ave_Loss {:.4f}  Ave_accu {:.4f}'.format(np.mean(ave_loss), np.mean(ave_accu)))
            print('\n')
        
            mean_accu = np.mean(ave_accu)
            
            if mean_accu > best:
                best = mean_accu
                sentence = translate(senten, trainer.model, 150)
                print('---生成结果---')
                print(sentence)
                print('------')
                print('saving_model')
                print('--------')
                self.checkpoint.save(self.checkpoint_dir)

trainer = Trainer(
    model=transformer,
    dataset=dataset,
    optimizer=optimizer,
    batch_size=GLOBAL_BATCH_SIZE,
    vocab_size=BPE_VOCAB_SIZE,
    epoch=EPOCHS,
)

trainer.single_gpu_train()