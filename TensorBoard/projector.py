# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector

import mnist_inference

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allocator_type = 'BFC'

LOG_DIR = (r'D:\codes\python\tensorflow\tensorflow_learning'
           r'\TensorBoard\log\projector_log')
DATA_SET_PATH = (r"D:\codes\python\tensorflow\tensorflow_learning"
                 r"\datasets\MNIST_data")
SPRITE_FILE = 'mnist_sprite.jpg'
META_FILE = 'mnist_meta.tsv'

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99

TENSOR_NAME = "FINAL_LOGITS"


def create_sprite_image(images):
    """
    使用给出的MNIST图片列表生成sprite图像
    :param images:
    :return:
    """
    if isinstance(images, list):
        images = np.array(images)

    img_h = images.shape[1]
    img_w = images.shape[2]

    # sprite图像时所有小图像拼成的大正方形矩阵，大正方形矩阵中每一个元素就是原来
    # 的小图像。因此，大正方形边长为sqrt(n)，n为小图像数量
    m = int(np.ceil(np.sqrt(images.shape[0])))

    # 使用全1来初始化最终的大图像
    sprite_image = np.ones((img_h*m, img_w*m))

    for i in range(m):
        for j in range(m):
            cur = i * m + j
            if cur < images.shape[0]:
                # 将当前小图像的内容复制到最终的sprite图像
                sprite_image[i*img_h:(i+1)*img_h, j*img_w:(j+1)*img_w] =\
                    images[cur]

    return sprite_image


def prepare():
    # one_hot为False，则得到的labels就是一个数字标签
    mnist = input_data.read_data_sets(DATA_SET_PATH, one_hot=False)

    # 生成sprite图像
    to_visualise = 1 - np.reshape(mnist.test.images, (-1, 28, 28))
    sprite_image = create_sprite_image(to_visualise)

    # 将生成的sprite图像放到相应的日志目录下
    path_for_mnist_sprites = os.path.join(LOG_DIR, SPRITE_FILE)
    plt.imsave(path_for_mnist_sprites, sprite_image, cmap='gray')
    plt.imshow(sprite_image, cmap='gray')

    # 生成每张图片对应的标签文件并写道相应的日志目录下
    path_for_mnist_metadata = os.path.join(LOG_DIR, META_FILE)
    with open(path_for_mnist_metadata, 'w') as f:
        f.write("Index\tLabel\n")
        for index, label in enumerate(mnist.test.labels):
            f.write("%d\t%d\n" % (index, label))


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
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y, labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 定义学习率、优化方法及每一轮执行训练的操作的命名空间。
    with tf.name_scope("train_step"):
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE, global_step,
            mnist.train.num_examples / BATCH_SIZE,
            LEARNING_RATE_DECAY, staircase=True)

        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
            loss, global_step=global_step)

        with tf.control_dependencies([train_step, variables_averages_op]):
            train_op = tf.no_op(name='train')

    # 训练模型。
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: xs, y_: ys})

            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch"
                      " is %g." % (i, loss_value))
        final_result = sess.run(y, feed_dict={x: mnist.test.images})

    return final_result


def visualisation(final_result):
    y = tf.Variable(final_result, name=TENSOR_NAME)
    summary_writer = tf.summary.FileWriter(LOG_DIR)

    proj_config = projector.ProjectorConfig()
    embedding = proj_config.embeddings.add()
    embedding.tensor_name = y.name

    # Specify where you find the metadata
    embedding.metadata_path = META_FILE

    # Specify where you find the sprite (we will create this later)
    embedding.sprite.image_path = SPRITE_FILE
    embedding.sprite.single_image_dim.extend([28, 28])

    # Say that you want to visualise the embeddings
    projector.visualize_embeddings(summary_writer, proj_config)

    sess = tf.InteractiveSession(config=config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(LOG_DIR, "model"), TRAINING_STEPS)

    summary_writer.close()


def main(argv=None):
    mnist = input_data.read_data_sets("../../datasets/MNIST_data", one_hot=True)
    final_result = train(mnist)
    visualisation(final_result)


if __name__ == '__main__':
    # prepare()
    main()

