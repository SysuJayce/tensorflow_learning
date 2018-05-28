# -*- coding: utf-8 -*-
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allocator_type = 'BFC'


def demo():
    # 定义一个简单的计算图，实现向量加法操作
    with tf.name_scope('input1'):
        input1 = tf.constant([1.0, 2.0, 3.0], name='input1')

    with tf.name_scope('input2'):
        input2 = tf.Variable(tf.random_uniform([3]), name='input2')

    output = tf.add_n([input1, input2], name='add')

    # 生成一个写日志的writer，并将当前的TensorFlow计算图写入日志。
    log_path = (r'D:\codes\python\tensorflow\tensorflow_learning'
                r'\TensorBoard\log')
    writer = tf.summary.FileWriter(log_path, tf.get_default_graph())
    writer.close()


if __name__ == '__main__':
    demo()
