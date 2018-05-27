# -*- coding:utf-8 -*-
import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 加载LeNet5_inference.py中定义的常量和前向传播的函数
import LeNet5_inference

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333,
                            allow_growth=True, allocator_type='BFC')

# 配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

# 模型保存在路径和文件名
MODEL_SAVE_PATH = r"D:\codes\python\tensorflow\book\CNN\model"
MODEL_NAME = "model.ckpt"
DATA_SET_PATH = (r"D:\codes\python\tensorflow\tensorflow-tutorial"
                 r"\Deep_Learning_with_TensorFlow\datasets\MNIST_data")


def train(mnist):
    x = tf.placeholder(tf.float32, [BATCH_SIZE, LeNet5_inference.IMAGE_SIZE,
                                    LeNet5_inference.IMAGE_SIZE,
                                    LeNet5_inference.NUM_CHANNELS],
                       'x-input')
    y_ = tf.placeholder(tf.float32, [None, LeNet5_inference.OUTPUT_NODE],
                        'y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # 直接使用LeNet5_inference.py中定义的前向传播过程
    y = LeNet5_inference.inference(x, False, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程
    # 滑动平均用于控制参数变化速率
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,
                                                          global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # tf.argmax(input, axis=None): axis中0表示列，1表示行
    # 因此这里labels的输入是一个nx1的向量，每一行代表正确值，logits表示预测值
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y,
        labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 总损失等于交叉熵 加上 每一层权重的L2正则化损失
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    # LEARNING_RATE_BASE：初始学习率
    # global_step：当前处于第几次迭代
    # decay_steps = mnist.train.num_examples / BATCH_SIZE：衰减周期(每多少次迭代
    # 衰减一次)
    # LEARNING_RATE_DECAY：衰减率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY)
    # 使用梯度下降优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss, global_step)
    # tf.control_dependencies(control_inputs)
    # control_inputs是一个由操作op或者张量tensor组成的列表。
    # 在tf.control_dependencies上下文管理器中的操作必须在control_inputs执行后执行
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化TensorFlow持久化类
    saver = tf.train.Saver(max_to_keep=10)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        tf.global_variables_initializer().run()

        # 在训练过程中不再测试模型在验证数据上的表现，验证和测试的过程将会由一个
        # 独立的程序来完成
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (BATCH_SIZE,
                                          LeNet5_inference.IMAGE_SIZE,
                                          LeNet5_inference.IMAGE_SIZE,
                                          LeNet5_inference.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: reshaped_xs, y_: ys})

            if i % 1000 == 0:
                print("After %d training step(s), loss on training "
                      "batch is %g." % (step, loss_value))

                # 保存当前的模型，注意这里给出了global_step参数，这样可以让每个
                # 被保存模型的文件名末尾加上训练的轮数， 比如“model.ckpt-1000”
                # global_step参数会被附加到checkpoint name中
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                           global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets(DATA_SET_PATH, one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
