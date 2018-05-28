import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)


def cnn():
    # 定义过滤器的权重
    filter_weight = tf.get_variable(
        'weights', [5, 3, 3, 16],
        initializer=tf.truncated_normal_initializer(stddev=0.1))

    # 定义bias变量
    biases = tf.get_variable('biases', [16],
                             initializer=tf.constant_initializer(0.1))

    # 调用tf.nn.conv2d进行cnn的前向传播
    conv = tf.nn.conv2d(input, filter_weight, [1, 1, 1, 1], 'SAME')

    # 为每一个节点添加bias值
    bias = tf.nn.bias_add(conv, biases)

    # 调用ReLu激活函数去线性化
    actived_conv = tf.nn.relu(bias)

    # 使用最大池化层进行前向传播
    pool = tf.nn.max_pool(actived_conv, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')


if __name__ == '__main__':
    pass
