# -*- coding: utf-8 -*-
import os
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, shape=[None, mnist_inference.INPUT_NODE],
                           name='x-input')
        y_ = tf.placeholder(tf.float32,
                            shape=[None, mnist_inference.OUTPUT_NODE],
                            name='y-input')
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}

        # 使用封装好的预测函数来计算前向传播结果。
        # 因为测试时不关注正则化损失的值，所以正则化损失函数设为None
        y = mnist_inference.inference(x, None)

        # 计算准确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 定义滑动平均变量
        variable_averages = tf.train.ExponentialMovingAverage(
            mnist_train.MOVING_AVERAGE_DECAY)
        # 使用variables_to_restore()来加载滑动平均变量的影子变量而非变量本身，
        # 将保存的模型的参数加载回程序
        # 将返回一个map，如gamma/ExponentialMovingAverage: gamma
        # gamma/ExponentialMovingAverage就是一个影子变量
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # 每隔EVAL_INTERVAL_SECS秒调用一次计算准确率过程
        while True:
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))\
                    as sess:
                ckpt = tf.train.get_checkpoint_state(
                    mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 获取迭代轮数
                    global_step = ckpt.model_checkpoint_path\
                                        .split('\\')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation "
                          "accuracy = %g" % (global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets(mnist_train.DATA_SET_PATH, one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()
