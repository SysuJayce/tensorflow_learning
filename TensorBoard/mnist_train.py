# -*- coding:utf-8 -*-
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载mnist_inference.py中定义的常量和前向传播的函数
import mnist_inference

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allocator_type = 'BFC'

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

# 配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

# 模型保存在路径和文件名
MODEL_SAVE_PATH = (r"D:\codes\python\tensorflow\tensorflow_learning"
                   r"\TensorBoard\MNIST\model")
MODEL_NAME = "model.ckpt"
DATA_SET_PATH = (r"D:\codes\python\tensorflow\tensorflow_learning"
                 r"\datasets\MNIST_data")

# 生成一个写日志的writer，并将当前的TensorFlow计算图写入日志。
log_path = (r'D:\codes\python\tensorflow\tensorflow_learning'
            r'\TensorBoard\log')


def train(mnist):
    #  输入数据的命名空间。
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE],
                           name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE],
                            name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 处理滑动平均的命名空间。
    with tf.name_scope("moving_average"):
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())

    # 计算损失函数的命名空间。
    with tf.name_scope("loss_function"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,
            labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 定义学习率、优化方法及每一轮执行训练的操作的命名空间。
    with tf.name_scope("train_step"):
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
            global_step, mnist.train.num_examples / BATCH_SIZE,
            LEARNING_RATE_DECAY, staircase=True)

        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
            loss, global_step=global_step)

        with tf.control_dependencies([train_step, variables_averages_op]):
            train_op = tf.no_op(name='train')

    writer = tf.summary.FileWriter(log_path, tf.get_default_graph())

    # 训练模型。
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            if i % 1000 == 0:
                # 配置运行时需要记录的信息。
                run_options = tf.RunOptions(
                    trace_level=tf.RunOptions.FULL_TRACE)
                # 运行时记录运行信息的proto。
                run_metadata = tf.RunMetadata()
                # 将配置信息和记录运行信息的proto传入运行的过程，从而记录每一个
                # 节点的时间、空间开销信息
                _, loss_value, step = sess.run([train_op, loss, global_step],
                                               feed_dict={x: xs, y_: ys},
                                               options=run_options,
                                               run_metadata=run_metadata)
                # 将节点在运行时的信息写入日志文件
                writer.add_run_metadata(run_metadata=run_metadata,
                                        tag=("tag%d" % i), global_step=i)
                print("After %d training step(s), loss on training batch"
                      " is %g." % (step, loss_value))
            else:
                _, loss_value, step = sess.run([train_op, loss, global_step],
                                               feed_dict={x: xs, y_: ys})

    writer.close()


def main(argv=None):
    mnist = input_data.read_data_sets(DATA_SET_PATH, one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
