# -*- coding: utf-8 -*-
import os
import glob
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 即使禁用显卡，使用内存，也还是内存不足。
# Process finished with exit code -1073740791 (0xC0000409)
# 所以这个inception-v3程序可以放弃了
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allocator_type = 'BFC'

INPUT_DATA = r'D:\codes\python\tensorflow\book\migration\flower_photos'
OUTPUT_FILE = (r'D:\codes\python\tensorflow\book\migration'
               r'\flower_processes_data.npy')

# 需要划分的验证、测试数据比例
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10


def create_image_lists(sess, testing_percentage, validation_percentage):
    """
    读取数据并划分成训练数据、验证数据和测试数据
    :param sess:
    :param testing_percentage:
    :param validation_percentage:
    :return:
    """

    # 获取所有子目录，包括了根目录
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    # 设置一个flag，用于跳过根目录的文件获取
    is_root_dir = True

    # 初始化各个数据集
    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []
    validation_images = []
    validation_labels = []
    current_label = 0

    # 读取所有子目录，以获取其下的图片信息
    for sub_dir in sub_dirs:
        # 跳过根目录
        if is_root_dir:
            is_root_dir = False
            continue

        # 获取子目录下所有图片文件
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []

        # 用basename获取最后一级目录名
        dir_name = os.path.basename(sub_dir)

        # 依次提取jpg, jpeg, JPG, JPEG格式的图片
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)

            # glob.glob(pattern)函数可以把符合pattern中的正则表达式的文件路径集
            # 合成一个列表。如当pattern为'/home/test/*.jpg'时可以提取/home/test
            # 目录下所有后缀为.jpg的文件路径，返回一个路径列表
            file_list.extend(glob.glob(file_glob))
            if not file_list:
                continue

            # 处理图片数据
            for file_name in file_list:
                # 读取并解析图片，将图片转化为299x299以便inception-v3模型来处理
                image_raw_data = gfile.FastGFile(file_name, 'rb').read()
                image = tf.image.decode_jpeg(image_raw_data)
                if image.dtype != tf.float32:
                    image = tf.image.convert_image_dtype(image, tf.float32)

                image = tf.image.resize_images(image, [299, 299])
                image_value = sess.run(image)

                # 随机划分数据集
                chance = np.random.randint(100)
                if chance < validation_percentage:
                    validation_images.append(image_value)
                    validation_labels.append(current_label)
                elif chance < validation_percentage + testing_percentage:
                    testing_images.append(image_value)
                    testing_labels.append(current_label)
                else:
                    training_images.append(image_value)
                    training_labels.append(current_label)

            current_label += 1

    # 将训练数据随机打乱以取得更好的训练效果
    # numpy.random.get_state()返回的结果中保存了在下一次shuffle时的序列生成方法
    # 通过set_state使得在shuffle了image之后能为label生成同样的随机序列
    # 这样就可以保证既打乱了数据顺序，又保持了image和label的一一对应关系
    state = np.random.get_state()
    np.random.shuffle(training_images)
    np.random.set_state(state)
    np.random.shuffle(training_labels)

    return np.asarray([training_images, training_labels,
                       validation_images, validation_labels,
                       testing_images, testing_labels])


def main():
    with tf.Session(config=config) as sess:
        processed_data = create_image_lists(
            sess, TEST_PERCENTAGE, VALIDATION_PERCENTAGE)

        # 通过numpy格式保存处理后的数据
        np.save(OUTPUT_FILE, processed_data)


if __name__ == '__main__':
    main()
