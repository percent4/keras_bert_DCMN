# -*- coding: utf-8 -*-
# @Time : 2021/4/3 16:45
# @Author : Jclian91
# @File : matrix_test.py
# @Place : Yangpu, Shanghai
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Add, Activation, Multiply, Lambda
from keras.optimizers import SGD

x1 = Input(shape=(768, ))
x2 = Input(shape=(768, ))
vector1 = Dense(768, input_dim=768)(x1)
vector2 = Dense(768, input_dim=768)(x2)
sigmoid_value = Activation(activation="sigmoid")(Add()([vector1, vector2]))
tmp1 = Multiply()([sigmoid_value, x1])
tmp2 = Multiply()([Lambda(lambda x: 1-x)(sigmoid_value), x2])
output = Add()([tmp1, tmp2])


model = Model([x1, x2], output)
model.summary()

test_x1 = np.random.randint(-5, 5, (32, 768))
test_x2 = np.random.randint(-5, 5, (32, 768))
test_y = (test_x1 + test_x2)/2
model.compile(loss='mse',
              optimizer=SGD(lr=1e-2))
model.fit([test_x1, test_x2], test_y, batch_size=1, epochs=100)

model.save("gate_mechaism_test.h5")