# -*- coding: utf-8 -*-
import tensorflow as tf

# 定义神经网络结构相关参数(各层节点个数)
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


def get_weight_variable(shape, regularizer):
    """
    通过tf.get_variable函数来获取变量。在训练神经网络时会创建这些变量；在测试
    时会通过保存的模型加载这些变量的取值。而且更加方便的是，因为可以在变量加载
    时将滑动平均变量重命名，所以可以直接通过同样的名字在训练时使用变量自身，而
    在测试时使用变量的滑动平均值。在这个函数中也会将变量的正则化损失加入损失集合
    """
    weights = tf.get_variable(
        "weights", shape,
        initializer=tf.truncated_normal_initializer(stddev=0.1))

    # 如果提供了正则化生成函数就将其加入losses集合中
    # 这是自定义的集合，不在TensorFlow自动管理的集合列表中
    if regularizer is not None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights


# 定义神经网络的前向传播过程
def inference(input_tensor, regularizer):
    # 声明第一层神经网络(这里是隐藏层)的变量并完成前向传播过程
    with tf.variable_scope('layer1'):
        weights = get_weight_variable(
            [INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE],
                                 initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    # 声明第二层神经网络(这里是输出层)的变量并完成前向传播过程
    with tf.variable_scope('layer2'):
        weights = get_weight_variable(
            [LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE],
                                 initializer=tf.constant_initializer(0.0))
        # 这里不需要使用激活函数，因为softmax在计算损失函数的时候会调用，在
        # MNIST数据集中，由于是选取输出节点中值最大的下标作为输出，所以用不用
        # softmax转换成概率都没影响。因此不需要用到激活函数。
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2
