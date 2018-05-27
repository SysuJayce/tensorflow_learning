# -*- coding: utf-8 -*-
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

# 配置神经网络参数
INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# 卷积层的尺寸和深度
CONV1_DEPTH = 32
CONV1_SIZE = 5
CONV2_DEPTH = 64
CONV2_SIZE = 5

# 全连接层的节点个数
FC_SIZE = 512


def inference(input_tensor, train, regularizer):
    """
    定义CNN的前向传播过程

    :param input_tensor: 输入张量
    :param train: 用于区分训练过程和测试过程
    :param regularizer: 正则化损失函数
    :return: 返回输出层的结果
    """
    # 声明第一层卷积层的变量并实现前向传播过程
    # 和标准LeNet-5模型不一样，这里定义的卷积层输入为28x28x1的矩阵，且使用全0
    # 填充，所以输出为28x28x32的矩阵
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            'weight', [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEPTH],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable(
            'bias', [CONV1_DEPTH], initializer=tf.constant_initializer(0.0))

        # 使用边长5，深度32的滤波器，步长为1，使用zero padding
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, [1, 1, 1, 1], 'SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 实现第二层池化层的前向传播过程。
    # 这里选用最大池化层，滤波器边长为2，步长为2，使用zero padding
    # 输入为28x28x32的矩阵，输出为14x14x32的矩阵
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    # 声明第三层卷积层的变量并实现前向传播。
    # 输入为14x14x32的矩阵，输出为14x14x64的矩阵
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable(
            'weight', [CONV2_SIZE, CONV2_SIZE, CONV1_DEPTH, CONV2_DEPTH],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable(
            'bias', [CONV2_DEPTH], initializer=tf.constant_initializer(0.0))

        # 使用边长5，深度64的滤波器，步长为1，使用zero padding
        conv2 = tf.nn.conv2d(pool1, conv2_weights, [1, 1, 1, 1], 'SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 实现第四层池化层的前向传播过程。
    # 这里选用最大池化层，滤波器边长为2，步长为2，使用zero padding
    # 输入为14x14x64的矩阵，输出为7x7x64的矩阵
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')									

        # 将第四层池化层的输出转化为第五层卷积层(全连接层)的输入格式。
        # 第四层输出为7x7x64的矩阵，然而第五层全连接层需要的输入格式为向量，所以在
        # 这里需要将这个矩阵拉直成一个向量。pool2.get_shape函数可以得到矩阵维度
        # 注意：因为每一层神经网络的输入输出都为一个batch矩阵，所以这里得到的维度也
        # 包含了一个batch中数据的个数
        pool_shape = pool2.get_shape().as_list()

        # 计算将矩阵拉直成向量之后的长度，即矩阵的三维之积。
        # 注意：get_shape()得到的列表第一个元素是batch中数据的个数，应舍弃
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

        # 通过tf.reshape()函数将第四层的输出变成一个batch向量
        reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    # 声明第五层卷积层(全连接层)的变量并实现前向传播。
    # 输入向量长度3136，输出向量长度512
    # 引入dropout，在训练中随机将部分节点的输出改为0。dropout可以避免过拟合，
    # 一般只在全连接层而不是卷积层或池化层使用
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable(
            'weight', [nodes, FC_SIZE],
            initializer=tf.truncated_normal_initializer(stddev=0.1))

        # 只有全连接层的权重需要加入正则化
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc1_weights))

        fc1_biases = tf.get_variable(
            'bias', [FC_SIZE], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)

        # 如果是在训练中，则使用dropout随机改部分节点输出为0
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    # 声明第六层全连接层的变量并实现前向传播
    # 这一层的输出通过Softmax之后就得到最后的分类结果
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable(
            'weight', [FC_SIZE, NUM_LABELS],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc2_weights))

        fc2_biases = tf.get_variable('bias', [NUM_LABELS],
                                     initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    # 返回第六层的输出，退出函数
    return logit
