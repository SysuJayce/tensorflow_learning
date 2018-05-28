# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np

import matplotlib as mpl
mpl.use('Agg')

from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allocator_type = 'BFC'


def rnn_propagation_example():
    # 定义输入和初始状态
    X = [1, 2]
    state = [0.0, 0.0]

    # 分别定义输入部分和前一状态部分的权重
    weight_cell_input = np.asarray([0.5, 0.6])
    weight_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])

    # 定义循环体内的bias
    bias_cell = np.asarray([0.1, -0.1])

    # 定义输出部分的全连接层的权重以及bias
    weight_output = np.asarray([[1.0], [2.0]])
    bias_output = 0.1

    # 按照时间顺序执行RNN的前向传播过程
    for i in range(len(X)):
        before_activation = np.dot(state, weight_cell_state) +\
                            X[i] * weight_cell_input + bias_cell
        state = np.tanh(before_activation)
        final_output = np.dot(state, weight_output) + bias_output
        print("before activation: ", before_activation)
        print("state: ", state)
        print("output: ", final_output)


if __name__ == '__main__':
    rnn_propagation_example()
