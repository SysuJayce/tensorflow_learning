# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# config = tf.ConfigProto(allow_soft_placement=True)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allocator_type = 'BFC'

DATA_SET_PATH = (r"D:\codes\python\tensorflow\tensorflow-tutorial"
                 r"\Deep_Learning_with_TensorFlow\datasets\MNIST_data")

# 输出TFRecord文件的地址
TFRECORD_PATH = (r'D:\codes\python\tensorflow\tensorflow_learning'
                 r'\TFRecord\out.tfrecords')


def _int64_feature(value):
    """
    生成整数型的属性
    :param value:
    :return:
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """
    生成字符串型的属性
    :param value:
    :return:
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_TFRecord():
    mnist = input_data.read_data_sets(DATA_SET_PATH, dtype=tf.uint8, one_hot=True)
    images = mnist.train.images
    labels = mnist.train.labels
    pixels = images.shape[1]
    num_example = mnist.train.num_examples

    # 创建一个writer来写TFRecord文件
    writer = tf.python_io.TFRecordWriter(TFRECORD_PATH)

    for index in range(num_example):
        # 将图像矩阵转化为一个字符串
        image_raw = images[index].tostring()

        # 将一个阳历转化为Example Protocol Buffer，并将所有信息写入这个数据结构
        example = tf.train.Example(features=tf.train.Features(feature={
            'pixels': _int64_feature(pixels),
            'label': _int64_feature(np.argmax(labels[index])),
            'image_raw': _bytes_feature(image_raw)}))

        writer.write(example.SerializeToString())

    writer.close()


def load_TFRecord():
    reader = tf.TFRecordReader()

    # 创建一个队列来维护输入文件列表
    filename_queue = tf.train.string_input_producer([TFRECORD_PATH])

    # 从文件中读出一个样例，也可以使用read_up_to函数一次性读取多个样例。
    _, serialized_example = reader.read(filename_queue)

    # 解析读入的一个样例，如果需要解析多个样例，可以使用parse_example函数
    features = tf.parse_single_example(
        serialized_example,

        # TensorFlow提供两种不同的属性解析方法。一种是tf.FixedLenFeature，这种方
        # 法解析的结果为一个Tensor。另一种方法时tf.VarLenFeature，这种方法解析的
        # 结果为SparseTensor，用于处理稀疏数据。这里解析数据的格式需要和写入数据
        # 时格式一致
        features={'image_raw': tf.FixedLenFeature([], tf.string),
                  'pixels': tf.FixedLenFeature([], tf.int64),
                  'label': tf.FixedLenFeature([], tf.int64)})

    # tf.decode_raw可以将字符串解析成图像对应的像素数组
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    pixels = tf.cast(features['pixels'], tf.int64)

    with tf.Session(config=config) as sess:

        # 启动多线程处理输入数据
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        # 每次运行可以读取TFRecord文件的一个样例。当所有样例都读完之后，会从头再读取
        for i in range(10):
            print(sess.run([image, label, pixels]))


def encode_decode_img():
    image_raw_data = tf.gfile.FastGFile(
        (r'D:\codes\python\tensorflow\tensorflow_learning'
         r'\image_process\input.jpg'), 'rb').read()

    with tf.Session(config=config) as sess:
        img_data = tf.image.decode_jpeg(image_raw_data)
        print(img_data.eval())

        plt.imshow(img_data.eval())
        plt.show()

        encoded_image = tf.image.encode_jpeg(img_data)
        with tf.gfile.GFile((r'D:\codes\python\tensorflow\tensorflow_learning'
                             r'\image_process\output.jpg'), 'wb') as f:
            f.write(encoded_image.eval())


def resize_image():
    image_raw_data = tf.gfile.FastGFile(
        r'D:\codes\python\tensorflow\tensorflow_learning\image_process\5003.JPG', 'rb').read()

    with tf.Session(config=config) as sess:
        img_data = tf.image.decode_jpeg(image_raw_data)
        # img_data = tf.image.convert_image_dtype(img_data, tf.float32)

        # resized0 = tf.image.resize_images(img_data, [300, 300], method=0)
        # plt.imshow(resized0.eval())
        # plt.show()
        #
        # resized1 = tf.image.resize_images(img_data, [300, 300], method=1)
        # plt.imshow(resized1.eval())
        # plt.show()
        #
        # resized2 = tf.image.resize_images(img_data, [300, 300], method=2)
        # plt.imshow(resized2.eval())
        # plt.show()
        #
        # resized3 = tf.image.resize_images(img_data, [300, 300], method=3)
        # plt.imshow(resized3.eval())
        # plt.show()

        # croped = tf.image.resize_image_with_crop_or_pad(img_data,
        #                                                 1000, 1000)
        # plt.imshow(croped.eval())
        # plt.show()
        #
        # croped2 = tf.image.resize_image_with_crop_or_pad(img_data, 3000, 3000)
        # plt.imshow(croped2.eval())
        # plt.show()

        # central_cropped = tf.image.central_crop(img_data, 0.5)
        # plt.imshow(central_cropped.eval())
        # plt.show()

        # flipped = tf.image.flip_up_down(img_data)
        # plt.imshow(flipped.eval())
        # plt.show()
        #
        # flipped = tf.image.flip_left_right(img_data)
        # plt.imshow(flipped.eval())
        # plt.show()
        #
        # flipped = tf.image.transpose_image(img_data)
        # plt.imshow(flipped.eval())
        # plt.show()

        # flipped = tf.image.random_flip_up_down(img_data)
        # plt.imshow(flipped.eval())
        # plt.show()
        #
        # flipped = tf.image.random_flip_left_right(img_data)
        # plt.imshow(flipped.eval())
        # plt.show()

        # adjusted = tf.image.adjust_brightness(img_data, -0.5)
        # adjusted = tf.clip_by_value(adjusted, 0.0, 1.0)
        # plt.imshow(adjusted.eval())
        # plt.show()
        #
        # adjusted = tf.image.adjust_brightness(img_data, 0.5)
        # adjusted = tf.clip_by_value(adjusted, 0.0, 1.0)
        # plt.imshow(adjusted.eval())
        # plt.show()
        #
        # adjusted = tf.image.adjust_brightness(img_data, 0.5)
        # adjusted = tf.clip_by_value(adjusted, 0.0, 1.0)
        # plt.imshow(adjusted.eval())
        # plt.show()

        img_data = tf.image.resize_images(img_data, [180, 267], method=1)
        boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
        # result = tf.image.draw_bounding_boxes(batched, boxes)
        begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
            tf.shape(img_data), boxes, min_object_covered=0.4)
        batched = tf.expand_dims(tf.image.convert_image_dtype(
            img_data, tf.float32), 0)
        image_with_box = tf.image.draw_bounding_boxes(batched, bbox_for_draw)
        plt.imshow(image_with_box.eval().reshape([180, 267, 3]))
        plt.show()
        distorted_image = tf.slice(img_data, begin, size)
        # plt.imshow(result.eval().reshape([180, 267, 3]))
        plt.imshow(distorted_image.eval())
        plt.show()


def tf_queue():
    q = tf.FIFOQueue(2, 'int32')
    init = q.enqueue_many(([0, 10],))

    x = q.dequeue()
    y = x + 1
    q_inc = q.enqueue([y])

    with tf.Session(config=config) as sess:
        sess.run(init)
        for _ in range(5):
            v, _ = sess.run([x, q_inc])
            print(v)


def multi_thread():
    import threading
    import time

    def my_loop(coord, worker_id):
        """
        在线程中运行，每隔1秒判断是否需要停止并打印自己的ID
        :param coord:
        :param worker_id:
        :return:
        """
        while not coord.should_stop():
            # 使用tf.Coordinator类提供的协同工具判断当前线程是否需要停止
            if np.random.rand() < 0.1:
                print("Stopping from id: %d" % worker_id)
                coord.request_stop()
            else:
                print("Working on id: %d" % worker_id)

            time.sleep(1)

    # 创建一个协同工具
    coord = tf.train.Coordinator()

    # 声明创建5个线程
    threads = [threading.Thread(
        target=my_loop, args=(coord, i, )) for i in range(5)]

    # 启动所有线程
    for t in threads: t.start()

    # 等待所有线程退出
    coord.join(threads)


def queue_runner():
    queue = tf.FIFOQueue(100, 'float')

    # 定义入队操作
    enqueue_op = queue.enqueue([tf.random_normal([1])])

    # 第一个参数为被操作队列，第二个参数为需要启动的线程数，每个线程都
    # 执行enqueue_op操作
    qr = tf.train.QueueRunner(queue, [enqueue_op] * 5)

    # 未指定集合时，默认加入tf.GraphKeys.QUEUE_RUNNERS
    tf.train.add_queue_runner(qr)

    # 定义出队操作
    out_tensor = queue.dequeue()

    with tf.Session(config=config) as sess:
        # 使用tf.train.Coordinator来协同启动的线程
        coord = tf.train.Coordinator()

        # 需要明确调用tf.train.start_queue_runners来启动所有线程。这个函数默认
        # 启动tf.GraphKeys.QUEUE_RUNNERS集合中所有的QueueRunner，但也可启动指定
        # 集合中的QueueRunner。因此，start_queue_runners和add_queue_runner指定的
        # 集合要相同
        threads = tf.train.start_queue_runners(sess, coord)

        # 获取队列中的取值
        for _ in range(3):
            print(sess.run(out_tensor)[0])

        # 使用Coordinator来停止所有线程
        coord.request_stop()
        coord.join(threads)


def test():
    num_shard = 2 # 写入文件个数
    instance_per_shard = 2 # 每个文件写入样例个数
    for i in range(num_shard):
        # 文件名格式为 0000n-of-0000m。n表示当前编号，m表示总数
        file_name = (r'D:\codes\python\tensorflow\tensorflow_learning\TFRecord'
                     r'\data.tfrecords-%.5d-of-%.5d' % (i, num_shard))
        writer = tf.python_io.TFRecordWriter(file_name)
        for j in range(instance_per_shard):
            example = tf.train.Example(features=tf.train.Features(feature={
                'i': _int64_feature(i), 'j': _int64_feature(j)}))
            writer.write(example.SerializeToString())
    writer.close()


def file_queue():
    files = tf.train.match_filenames_once(r'D:\codes\python\tensorflow\tensorflow_learning'
                                          r'\TFRecord\data.tfrecords-*')
    filename_queue = tf.train.string_input_producer(files, num_epochs=1)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        'i': tf.FixedLenFeature([], tf.int64),
        'j': tf.FixedLenFeature([], tf.int64)})

    with tf.Session(config=config) as sess:
        tf.local_variables_initializer().run()
        print(sess.run(files))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        for i in range(6):
            print(sess.run([features['i'], features['j']]))

        coord.request_stop()
        coord.join(threads)


def gen_batch():
    files = tf.train.match_filenames_once(r'D:\codes\python\tensorflow\tensorflow_learning'
                                          r'\TFRecord\data.tfrecords-*')
    filename_queue = tf.train.string_input_producer(files, num_epochs=1)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        'i': tf.FixedLenFeature([], tf.int64),
        'j': tf.FixedLenFeature([], tf.int64)})

    example, label = features['i'], features['j']
    batch_size = 2
    capacity = 1000 + batch_size * 3

    example_batch, label_batch = tf.train.batch([example, label], batch_size,
                                                capacity=capacity)

    with tf.Session(config=config) as sess:
        # tf.initialize_all_variables()这个函数已经被弃用，所以只能分别初始化
        # global和local变量
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        for i in range(2):
            cur_example_batch, cur_label_batch = sess.run(
                [example_batch, label_batch])
            print(cur_example_batch, cur_label_batch)

        coord.request_stop()
        coord.join(threads)


def data_set():
    input_data = [1, 2, 3, 4, 5]
    dataset = tf.data.Dataset.from_tensor_slices(input_data)
    iterator = dataset.make_one_shot_iterator()
    x = iterator.get_next()
    y = x * x

    with tf.Session(config=config) as sess:
        for i in range(len(input_data)):
            print(sess.run(y))


def data_set_from_tfrecord():

    def parser(record):
        features = tf.parse_single_example(record, features={
            'feat1': tf.FixedLenFeature([], tf.int64),
            'feat2': tf.FixedLenFeature([], tf.int64)})
        return features['feat1'], features['feat2']

    input_files = [r'D:\codes\python\tensorflow\tensorflow_learning\TFRecord\data.tfrecords-00000-of-00002', r'D:\codes\python\tensorflow\tensorflow_learning\TFRecord\data.tfrecords-00001-of-00002']
    dataset = tf.data.TFRecordDataset(input_files)

    # map函数表示对数据集中的每一条数据进行调用相应方法，使用TFRecordDataset读出
    # 的是二进制的数据，需要通过map来调用parser方法对二进制数据进行解析
    dataset = dataset.map(parser)

    iterator = dataset.make_one_shot_iterator()

    feat1, feat2 = iterator.get_next()

    with tf.Session(config=config) as sess:
        for i in range(10):
            f1, f2 = sess.run([feat1, feat2])
            # print(sess.run([feat1, feat2]))


def data_set_from_tfrecord_placeholder():

    def parser(record):
        features = tf.parse_single_example(record, features={
            'feat1': tf.FixedLenFeature([], tf.int64),
            'feat2': tf.FixedLenFeature([], tf.int64)})
        return features['feat1'], features['feat2']

    input_files = tf.placeholder(tf.string)
    # input_files = [r'D:\codes\python\tensorflow\tensorflow_learning\TFRecord\data.tfrecords-00000-of-00002', r'D:\codes\python\tensorflow\tensorflow_learning\TFRecord\data.tfrecords-00001-of-00002']
    dataset = tf.data.TFRecordDataset(input_files)

    # map函数表示对数据集中的每一条数据进行调用相应方法，使用TFRecordDataset读出
    # 的是二进制的数据，需要通过map来调用parser方法对二进制数据进行解析
    dataset = dataset.map(parser)

    # iterator = dataset.make_one_shot_iterator()
    iterator = dataset.make_initializable_iterator()

    feat1, feat2 = iterator.get_next()

    with tf.Session(config=config) as sess:
        sess.run(iterator.initializer, feed_dict={'input_files': [r'D:\codes\python\tensorflow\tensorflow_learning\TFRecord\data.tfrecords-00000-of-00002', r'D:\codes\python\tensorflow\tensorflow_learning\TFRecord\data.tfrecords-00001-of-00002']})
        while True:
            try:
                sess.run([feat1, feat2])
            except tf.errors.OutOfRangeError:
                break
        # for i in range(10):
        #     f1, f2 = sess.run([feat1, feat2])
            # print(sess.run([feat1, feat2]))


if __name__ == '__main__':
    # load_TFRecord()
    # encode_decode_img()
    # resize_image()
    # tf_queue()
    # multi_thread()
    # queue_runner()
    # test()
    # file_queue()
    # gen_batch()
    # data_set()
    # data_set_from_tfrecord()
    data_set_from_tfrecord_placeholder()
