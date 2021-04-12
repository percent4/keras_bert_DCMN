# -*- coding: utf-8 -*-
# @Time : 2021/4/3 17:37
# @Author : Jclian91
# @File : model_train.py
# @Place : Yangpu, Shanghai
import numpy as np
from keras_bert import Tokenizer
from keras.losses import categorical_crossentropy
from keras_bert import AdamWarmup, calc_train_steps

from load_data import train_samples, dev_samples
from model import SimpleMultiChoiceMRC
from params import (dataset,
                    VOCAB_FILE_PATH,
                    BATCH_SIZE,
                    CHECKPOINT_FILE_PATH,
                    CONFIG_FILE_PATH,
                    NUM_CHOICES,
                    WARMUP_RATION,
                    EPOCH,
                    MAX_P_LENGTH,
                    MAX_Q_LENGTH,
                    MAX_A_LENGTH
                    )

token_dict = {}
with open(VOCAB_FILE_PATH, 'r', encoding='utf-8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

tokenizer = Tokenizer(token_dict, cased=False)


class DataGenerator:

    def __init__(self, data, batch_size=BATCH_SIZE):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            P_X1 = np.empty(shape=(self.batch_size, NUM_CHOICES, MAX_P_LENGTH))
            P_X2 = np.empty(shape=(self.batch_size, NUM_CHOICES, MAX_P_LENGTH))
            Q_X1 = np.empty(shape=(self.batch_size, NUM_CHOICES, MAX_Q_LENGTH))
            Q_X2 = np.empty(shape=(self.batch_size, NUM_CHOICES, MAX_Q_LENGTH))
            A_X1 = np.empty(shape=(self.batch_size, NUM_CHOICES, MAX_A_LENGTH))
            A_X2 = np.empty(shape=(self.batch_size, NUM_CHOICES, MAX_A_LENGTH))
            Y = np.zeros(shape=(self.batch_size, NUM_CHOICES))
            for i in idxs:
                sample = self.data[i]
                Y[i % self.batch_size, sample.correct_answer] = 1
                for choice_num, answer in enumerate(sample.answers):
                    p_x1, p_x2 = tokenizer.encode(sample.article, max_len=MAX_P_LENGTH)
                    P_X1[i % self.batch_size, choice_num, :] = p_x1
                    P_X2[i % self.batch_size, choice_num, :] = p_x2
                    q_x1, q_x2 = tokenizer.encode(sample.question, max_len=MAX_Q_LENGTH)
                    Q_X1[i % self.batch_size, choice_num, :] = q_x1
                    Q_X2[i % self.batch_size, choice_num, :] = q_x2
                    a_x1, a_x2 = tokenizer.encode(answer, max_len=MAX_A_LENGTH)
                    A_X1[i % self.batch_size, choice_num, :] = a_x1
                    A_X2[i % self.batch_size, choice_num, :] = a_x2

                if ((i+1) % self.batch_size == 0) or i == idxs[-1]:
                    yield [P_X1, P_X2, Q_X1, Q_X2, A_X1, A_X2], Y
                    P_X1 = np.empty(shape=(self.batch_size, NUM_CHOICES, MAX_P_LENGTH))
                    P_X2 = np.empty(shape=(self.batch_size, NUM_CHOICES, MAX_P_LENGTH))
                    Q_X1 = np.empty(shape=(self.batch_size, NUM_CHOICES, MAX_Q_LENGTH))
                    Q_X2 = np.empty(shape=(self.batch_size, NUM_CHOICES, MAX_Q_LENGTH))
                    A_X1 = np.empty(shape=(self.batch_size, NUM_CHOICES, MAX_A_LENGTH))
                    A_X2 = np.empty(shape=(self.batch_size, NUM_CHOICES, MAX_A_LENGTH))
                    Y = np.zeros(shape=(self.batch_size, NUM_CHOICES))


if __name__ == '__main__':

    # 模型训练
    train_D = DataGenerator(train_samples)
    dev_D = DataGenerator(dev_samples)
    model = SimpleMultiChoiceMRC(CONFIG_FILE_PATH, CHECKPOINT_FILE_PATH, MAX_P_LENGTH,
                                 MAX_Q_LENGTH, MAX_A_LENGTH, NUM_CHOICES).create_model()
    # add warmup
    total_steps, warmup_steps = calc_train_steps(
        num_example=len(train_samples),
        batch_size=BATCH_SIZE,
        epochs=EPOCH,
        warmup_proportion=WARMUP_RATION,
    )
    optimizer = AdamWarmup(total_steps, warmup_steps, lr=1e-5, min_lr=1e-7)
    model.compile(
        loss=categorical_crossentropy,
        optimizer=optimizer,
        metrics=['accuracy']
    )

    print("begin model training...")
    model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=EPOCH,
        validation_data=dev_D.__iter__(),
        validation_steps=len(dev_D)
    )

    print("finish model training!")

    result = model.evaluate_generator(dev_D.__iter__(), steps=len(dev_D))
    print("model evaluate result: ", result)

    # 模型保存
    model.save_weights('DCMN_{}_{}.h5'.format(dataset, result[-1]))
    print("Model saved!")
