# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

# 加载通过TensorFlow-Slim定义好的inception-v3模型
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'

# 经过data_process 处理后的数据文件
INPUT_DATA = (r'D:\codes\python\tensorflow\book\migration'
              r'\flower_processes_data.npy')

TRAIN_FILE = r'D:\codes\python\tensorflow\book\migration'

CKPT_FILE = r'D:\codes\python\tensorflow\book\migration\inception_v3.ckpt'

LEARNING_RATE = 0.0001
STEPS = 300
BATCH = 32
N_CLASSES = 5

CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits, InceptionV3/AuxLogits'

TRAINABLE_SCOPES = 'InceptionV3/Logits, InceptionV3/AuxLogits'


def get_tuned_variables():
    """
    获取不需要训练，只需要加载的参数
    :return: 
    """
    exclusions = [scope.strip() for scope in
                  CHECKPOINT_EXCLUDE_SCOPES.split(',')]
    variables_to_restore = []

    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break

        if not excluded:
            variables_to_restore.append(var)

    return variables_to_restore


def get_trainable_variables():
    """
    获取需要训练的变量
    :return: 
    """
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(',')]
    variables_to_train = []

    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)

    return variables_to_train


def main():
    processed_data = np.load(INPUT_DATA)
    training_images = processed_data[0]
    n_training_example = len(training_images)
    training_labels = processed_data[1]
    validation_images = processed_data[2]
    validation_labels = processed_data[3]
    testing_images = processed_data[4]
    testing_labels = processed_data[5]
    print("%d training examples, %d validation examples and %d"
          "testing examples." % (
          n_training_example, len(validation_labels), len(testing_labels)))

    # 定义inception-v3的输入
    images = tf.placeholder(tf.float32, [None, 299, 299, 3], 'input_images')
    labels = tf.placeholder(tf.int64, [None], 'labels')

    # 定义inveption-v3的模型。直接从ckpt读取的只是参数值，所以需要定义一个对应模
    # 型。因为预先训练好的inception-v3模型中使用的batch normalization参数与新的
    # 数据会有差异，导致结果很差，所以这里直接使用同一个模型来进行测试
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(images, N_CLASSES)

    # 获取需要训练的变量
    trainable_variables = get_trainable_variables()

    # 定义交叉熵损失。注意在模型定义的时候已经将正则化损失加入损失集合了
    tf.losses.softmax_cross_entropy(tf.one_hot(labels, N_CLASSES), logits)

    # 定义训练过程，这里minimize的过程中指定了需要优化的变量集合
    train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(
        tf.losses.get_total_loss())

    # 计算正确率
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        evaluation_step = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32))

    # 定义加载模型的函数
    load_fn = slim.assign_from_checkpoint_fn(CKPT_FILE, get_tuned_variables(),
                                             True)

    # 定义保存新的训练好的模型的函数
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        # 在加载变量之前初始化所有变量，否则会把加载进来的变量重新赋值
        init = tf.global_variables_initializer()
        sess.run(init)

        # 加载已训练好的模型
        print("Loading tuned variables from %s" % CKPT_FILE)
        load_fn(sess)

        start = 0
        end = BATCH
        for i in range(STEPS):
            # 运行训练过程，这里不会更新全部的参数，只会更新指定的部分参数
            sess.run(train_step, {images: training_images[start:end],
                                  labels: training_labels[start:end]})

            # 输出日志
            if i % 30 == 0 or i + 1 == STEPS:
                saver.save(sess, TRAIN_FILE, i)
                validation_accuracy = sess.run(evaluation_step,
                                               {images: validation_images,
                                                labels: validation_labels})
                print("Step %d: Validation accuracy = %.1f%%" % (
                    i, validation_accuracy * 100.0))

            # 因为在数据预处理的时候已经打乱了数据，所以这里只需要顺序使用训练
            # 数据即可，不必使用next_batch
            start = end
            if start == n_training_example:
                start = 0
            end = min(start + BATCH, n_training_example)

        # 训练结束后再测试集上测试正确率
        test_accuracy = sess.run(evaluation_step, {images: testing_images,
                                                   labels: testing_labels})
        print("Final test accuracy = %.1f%%" % (test_accuracy * 100.0))


if __name__ == '__main__':
    tf.app.run()
