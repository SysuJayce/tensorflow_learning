# -*- coding:utf-8 -*-
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allocator_type = 'BFC'

SUMMARY_DIR = (r'D:\codes\python\tensorflow\tensorflow_learning'
               r'\TensorBoard\log\monitor')
DATA_SET_PATH = (r"D:\codes\python\tensorflow\tensorflow-tutorial"
                 r"\Deep_Learning_with_TensorFlow\datasets\MNIST_data")
BATCH_SIZE = 100
TRAIN_STEPS = 3000


def variable_summaries(var, name):
    """
    生成变量监控信息并定义生成监控信息日志的操作。
    var给出了需要记录的张量，name给出了在可视化结果中显示的图表名称
    这里主要是对均值和标准差进行日志记录
    :param var:
    :param name:
    :return:
    """
    # 将生成监控信息的操作放到同一个命名空间下
    with tf.name_scope('summaries'):
        tf.summary.histogram(name, var)
        # 计算变量的平均值，mean为命名空间
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        # 计算变量的标准差
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """
    生成一层全连接层神经网络，可以看作一个全连接层神经网络生成器(模板)
    :param input_tensor:
    :param input_dim:
    :param output_dim:
    :param layer_name:
    :param act:
    :return:
    """
    # 将同一层神经网络放在统一的命名空间里。
    # 这里分成三个命名空间，分别是weights, bias, 以及预测的结果的命名空间
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal([input_dim, output_dim],
                                                      stddev=0.1))
            variable_summaries(weights, layer_name + '/weights')

        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.0, shape=[output_dim]))
            variable_summaries(biases, layer_name + '/biases')

        with tf.name_scope('Wx_plus_b'):
            pre_activate = tf.matmul(input_tensor, weights) + biases
            # 记录神经网络输出节点在经过激活函数之前的分布
            tf.summary.histogram(layer_name + '/pre_activations', pre_activate)

        activations = act(pre_activate, name='activations')

        # 记录神经网络输出节点经过激活函数之后的分布
        tf.summary.histogram(layer_name + '/activations', activations)
        return activations


def main(_):
    mnist = input_data.read_data_sets(DATA_SET_PATH, one_hot=True)
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

    # 通过tf.summary定义日志的写操作
    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)

    # 生成隐藏层和输出层神经网络
    hidden1 = nn_layer(x, 784, 500, 'layer1')
    y = nn_layer(hidden1, 500, 10, 'layer2', act=tf.identity)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        tf.summary.scalar('cross entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    # 通过tf.summary定义的写操作需要通过sess.run来执行，为避免繁复操作，可以通过
    # 将这些写操作merge在一起，调用tf.summary.merge_all函数来整理所有的
    # 日志生成操作
    merged = tf.summary.merge_all()
    with tf.Session(config=config) as sess:
        # 初始化写日志的writer，并将当前TensorFlow计算图写入日志
        summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        tf.global_variables_initializer().run()

        for i in range(TRAIN_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            summary, _ = sess.run([merged, train_step],
                                  feed_dict={x: xs, y_: ys})
            summary_writer.add_summary(summary, i)

    summary_writer.close()


if __name__ == '__main__':
    tf.app.run()
