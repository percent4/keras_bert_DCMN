# -*- coding: utf-8 -*-
# @Time : 2021/4/3 17:01
# @Author : Jclian91
# @File : model.py
# @Place : Yangpu, Shanghai
# main architecture of SimpleMultiChoiceMRC
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Lambda, Dense, Activation, GlobalMaxPool1D, Add, Multiply, concatenate, Permute, Dot
from keras_bert import load_trained_model_from_checkpoint

HIDDLE_SIZE = 768


# model structure of SimpleMultiChoiceMRC
class SimpleMultiChoiceMRC(object):
    def __init__(self, config_path, checkpoint_path, p_max_len, q_max_len, a_max_len, num_choices):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.p_max_len = p_max_len
        self.q_max_len = q_max_len
        self.a_max_len = a_max_len
        self.num_choices = num_choices  

    def create_model(self):
        # BERT model
        bert_model = load_trained_model_from_checkpoint(self.config_path, self.checkpoint_path)
        for layer in bert_model.layers:
            layer.trainable = True

        # get bert encoder vector
        p_x1_in = Input(shape=(self.num_choices, self.p_max_len,), name="p_token_ids")
        p_x2_in = Input(shape=(self.num_choices, self.p_max_len,), name="p_segment_ids")
        q_x1_in = Input(shape=(self.num_choices, self.q_max_len,), name="q_token_ids")
        q_x2_in = Input(shape=(self.num_choices, self.q_max_len,), name="q_segment_ids")
        a_x1_in = Input(shape=(self.num_choices, self.a_max_len,), name="a_token_ids")
        a_x2_in = Input(shape=(self.num_choices, self.a_max_len,), name="a_segment_ids")
        p_reshape_x1_in = Lambda(lambda x: tf.reshape(x, [-1, self.p_max_len]), name="p_reshape1")(p_x1_in)
        p_reshape_x2_in = Lambda(lambda x: tf.reshape(x, [-1, self.p_max_len]), name="p_reshape2")(p_x2_in)
        q_reshape_x1_in = Lambda(lambda x: tf.reshape(x, [-1, self.q_max_len]), name="q_reshape1")(q_x1_in)
        q_reshape_x2_in = Lambda(lambda x: tf.reshape(x, [-1, self.q_max_len]), name="q_reshape2")(q_x2_in)
        a_reshape_x1_in = Lambda(lambda x: tf.reshape(x, [-1, self.a_max_len]), name="a_reshape1")(a_x1_in)
        a_reshape_x2_in = Lambda(lambda x: tf.reshape(x, [-1, self.a_max_len]), name="a_reshape2")(a_x2_in)

        # Get encode layer
        p_bert_layer = bert_model([p_reshape_x1_in, p_reshape_x2_in])
        p_encode_vector = Lambda(lambda x: x[:, :], name="p_encode_layer")(p_bert_layer)
        q_bert_layer = bert_model([q_reshape_x1_in, q_reshape_x2_in])
        q_encode_vector = Lambda(lambda x: x[:, :], name="q_encode_layer")(q_bert_layer)
        a_bert_layer = bert_model([a_reshape_x1_in, a_reshape_x2_in])
        a_encode_vector = Lambda(lambda x: x[:, :], name="a_encode_layer")(a_bert_layer)

        # Bidirectional matching
        p_q_s1, p_q_s2 = self.bidirectional_matching(p_encode_vector, q_encode_vector)
        p_a_s1, p_a_s2 = self.bidirectional_matching(p_encode_vector, a_encode_vector)
        q_a_s1, q_a_s2 = self.bidirectional_matching(q_encode_vector, a_encode_vector)

        # Gated Mechanism
        p_q_fusion = self.gated_mechanism(p_q_s1, p_q_s2)
        p_a_fusion = self.gated_mechanism(p_a_s1, p_a_s2)
        q_a_fusion = self.gated_mechanism(q_a_s1, q_a_s2)

        # concat layer
        concat_layer = concatenate([p_q_fusion, p_a_fusion, q_a_fusion])

        # classifier layer
        logits = Dense(1, name="classifier", activation=None)(concat_layer)
        reshape_layer = Lambda(lambda x: tf.reshape(x, [-1, self.num_choices]), name="reshape_cls")(logits)
        # log_softmax = Activation(activation=tf.nn.log_softmax)(reshape_layer)
        output = Activation(activation="softmax")(reshape_layer)

        # model
        model = Model([p_x1_in, p_x2_in, q_x1_in, q_x2_in, a_x1_in, a_x2_in], output)
        model.summary()

        return model

    @staticmethod
    def bidirectional_matching(h1, h2):
        # h1 size: p,l --- h1 size: a, l, l = HIDDLE_SIZE
        # vector size: p, l
        h1_w = Dense(HIDDLE_SIZE, input_dim=HIDDLE_SIZE)(h1)
        # vector size: p, a
        h1_w_h2 = Dot(axes=2)([h1_w, h2])
        g12 = Activation(activation="softmax")(h1_w_h2)
        # vector size: p, l
        e1 = Dot(axes=1)([Permute((2, 1))(g12), h2])
        # vector size: a, l
        e2 = Dot(axes=1)([g12, h1])
        s1 = Activation(activation="relu")(Dense(HIDDLE_SIZE, input_dim=HIDDLE_SIZE)(e1))
        s2 = Activation(activation="relu")(Dense(HIDDLE_SIZE, input_dim=HIDDLE_SIZE)(e2))
        return s1, s2

    @staticmethod
    def gated_mechanism(x1, x2):
        pooling_x1 = GlobalMaxPool1D()(x1)
        pooling_x2 = GlobalMaxPool1D()(x2)
        vector1 = Dense(HIDDLE_SIZE, input_dim=HIDDLE_SIZE)(pooling_x1)
        vector2 = Dense(HIDDLE_SIZE, input_dim=HIDDLE_SIZE)(pooling_x2)
        sigmoid_value = Activation(activation="sigmoid")(Add()([vector1, vector2]))
        tmp1 = Multiply()([sigmoid_value, pooling_x1])
        tmp2 = Multiply()([Lambda(lambda x: 1-x)(sigmoid_value), pooling_x2])
        gated_vector = Add()([tmp1, tmp2])
        return gated_vector


if __name__ == '__main__':
    model_config = "./chinese_L-12_H-768_A-12/bert_config.json"
    model_checkpoint = "./chinese_L-12_H-768_A-12/bert_model.ckpt"
    model = SimpleMultiChoiceMRC(model_config, model_checkpoint, 400, 32, 32, 4).create_model()