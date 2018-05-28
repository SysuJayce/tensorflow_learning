# -*- coding: utf-8 -*-
import keras
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, LSTM, Input
from keras.datasets import imdb, mnist


def sequential_model():
    # 最多使用的单词数
    max_features = 20000
    # 循环神经网络的截断长度
    maxlen = 80
    batch_size = 32

    # 加载数据并将单词转化为ID, max_features给出了最多使用的单词数。和自然语言模型
    # 类似，会将出现频率较低的单词替换为统一的ID。通过Keras封装的API会生成25000条
    # 训练数据和对应的测试数据，每一条数据可以被看成一段话，并且每段话都有一个好评
    # 或者差评的标签

    (trainX, trainY), (testX, testY) = imdb.load_data(num_words=max_features)
    print(len(trainX), 'train sequences')
    print(len(trainY), 'test sequences')

    # 由于在自然语言中，每一段话的长度不一定相同，但循环神经网络的循环长度是固定的，
    # 所以这里需要先将所有段落统一成固定长度。对于长度不够的段落，使用默认值0填充，
    # 对于超过长度的段落则直接忽略掉超过的部分
    trainX = sequence.pad_sequences(trainX, maxlen=maxlen)
    testX = sequence.pad_sequences(testX, maxlen=maxlen)
    print('trainX shape:', trainX.shape)
    print('trainY shape:', trainY.shape)

    # 在完成数据预处理之后构建模型
    model = Sequential()
    # 构建Embedding层。128代表Embedding层的向量维度
    model.add(Embedding(max_features, 128))
    # 构建LSTM层
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    # 构建最后的全连接层。注意在上面构建LSTM层时只会得到最后一个节点的输出，如果需要
    # 输出每个时间点的结果，那么可以将return_sequences参数设为True
    model.add(Dense(1, activation='sigmoid'))

    # 指定损失函数、优化函数和评测指标
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    # 指定训练数据、训练轮数、batch大小以及验证数据
    model.fit(trainX, trainY, batch_size=batch_size, epochs=15,
              validation_data=(testX, testY))

    # 在测试数据上评测模型
    score = model.evaluate(testX, testY, batch_size=batch_size)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


def inception_model():
    # 从MNIST数据集读取训练数据和测试数据
    (trainX, trainY), (testX, testY) = mnist.load_data()
    trainX = trainX.reshape(trainX.shape[0], 28 * 28)
    testX = testX.reshape(testX.shape[0], 28 * 28)

    trainX = trainX.astype('float32')
    testX = testX.astype('float32')
    trainX /= 255.0
    testX /= 255.0

    trainY = keras.utils.to_categorical(trainY, 10)
    testY = keras.utils.to_categorical(testY, 10)

    # 定义两个输入，一个是输入层1，另一个是输出层1。（图10-2）
    input1 = Input(shape=(784,), name='input1')
    input2 = Input(shape=(10,), name='input2')

    # x为只有一个隐藏节点的全连接网络。图10-2的隐藏层
    x = Dense(1, activation='relu')(input1)
    # 图10-2的输出层1的输出
    output1 = Dense(10, activation='softmax', name='output1')(x)
    # 把x和output1结合作为图10-2中输出层2的输入
    y = keras.layers.concatenate([x, input2])
    # 图10-2的输出层2的输出
    output2 = Dense(10, activation='softmax', name='output2')(y)

    # 定义一个如图10-2所示的模型。只需要给出输入和输出的参数就可以完成模型的构建
    model = Model(inputs=[input1, input2], outputs=[output1, output2])
    # 定义模型的损失函数、优化函数以及评测方法。在定义损失函数时可以分别为不同
    # 输出层定义不同的损失函数，然后再按需定义对应权重
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(),
                  loss_weights=[1, 0.1],
                  metrics=['accuracy'])

    # 训练模型，可以简单地按照输入、输出列表提供参数。但使用字典形式会更明确，
    # 可以避免输入、输出顺序不一致产生问题
    model.fit({'input1': trainX, 'input2': trainY},
              {'output1': trainY, 'output2': trainY},
              batch_size=128,
              epochs=20,
              validation_data=([testX, testY], [testY, testY]))


def combination():
    """
    结合Keras和原生态TensorFlow API
    :return:
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.allocator_type = 'BFC'
    DATA_SET_PATH = (r"D:\codes\python\tensorflow\tensorflow_learning"
                     r"\datasets\MNIST_data")
    mnist_data = input_data.read_data_sets(DATA_SET_PATH, one_hot=True)

    x = tf.placeholder(tf.float32, shape=(None, 784))
    y_ = tf.placeholder(tf.float32, shape=(None, 10))

    # 使用tensorflow中的keras API定义网络模型
    net = tf.keras.layers.Dense(500, activation='relu')(x)
    y = tf.keras.layers.Dense(10, activation='softmax')(net)

    # 定义损失函数和优化函数以及评测方法
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_, y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    acc_value = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(y_, y))

    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()

        for i in range(10000):
            xs, ys = mnist_data.train.next_batch(100)
            _, loss_value = sess.run([train_step, loss],
                                     feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch"
                      "is %g." % (i, loss_value))

        print(acc_value.eval(feed_dict={x: mnist_data.test.images,
                                        y_: mnist_data.test.labels}))


def estimator_model():
    """
    使用Dataset作为Estimator的输入，并且自定义Estimator模型来完成MNIST数据集的
    训练和测试
    :return:
    """
    feature_names = ['SepalLength', 'SepalWidth',
                     'PetalLength', 'PetalWidth']

    tf.logging.set_verbosity(tf.logging.INFO)
    csv_path_root = (r'D:\codes\python\tensorflow\tensorflow_learning'
                     r'\datasets')

    def my_input_fn(file_path, perform_shuffle=False, repeat_count=1):
        """
        Estimator的自定义输入函数需要每一次被调用时可以得到一个batch的数据（包
        括特征和标签）。虽然自定义输入函数不能有参数，但是可以使用lambda表达式
        将自定义输入函数转化为不带参数的函数
        :param file_path:
        :param perform_shuffle:
        :param repeat_count:
        :return:
        """
        # 定义解析csv文件中一行的方法
        def decode_csv(line):
            # 将一行中的数据解析出来。前四列为特征，最后一列为标签
            parsed_line = tf.decode_csv(line, [[0.], [0.], [0.], [0.], [0]])
            # Estimator的输入函数要求特征一个字典，所以这里返回的也需要是一个
            # 字典。字典Key的定义需要和DNNClassifier中feature_columns的定义匹配
            # return {'x': tf.constant(parsed_line[:-1])}, parsed_line[-1:]
            label = parsed_line[-1]
            del parsed_line[-1]
            return dict(zip(feature_names, parsed_line)), label
            # label = parsed_line[-1]
            # del parsed_line[-1]
            # return {'x': parsed_line}, label

        # 使用数据集处理输入数据
        dataset = tf.data.TextLineDataset(file_path).skip(1).map(decode_csv)
        if perform_shuffle:
            dataset = dataset.shuffle(buffer_size=256)
        dataset = dataset.repeat(repeat_count)
        dataset = dataset.batch(32)
        iterator = dataset.make_one_shot_iterator()
        # 通过定义的数据集得到一个batch的输入数据
        batch_features, batch_labels = iterator.get_next()
        # 如果是为预测过程提供输入数据，那么batch_labels可以直接使用None
        return batch_features, batch_labels

    # feature_columns = [tf.feature_column.numeric_column('x', shape=[4])]
    feature_columns = [tf.feature_column.numeric_column(k) for k in
                       feature_names]
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 10],
                                            n_classes=3)

    # 使用lambda表达式将训练相关的信息传入自定义输入数据处理函数并生成Estimator
    # 需要的输入函数
    classifier.train(input_fn=lambda: my_input_fn(
        csv_path_root+'\\iris_training.csv', True, 10))

    # 生成测试需要的输入函数
    test_results = classifier.evaluate(input_fn=lambda: my_input_fn(
        csv_path_root+'\\iris_test.csv', False, 1))
    print("\nRest accuracy: %g%%" % (test_results['accuracy']*100))


if __name__ == '__main__':
    # inception_model()
    # combination()
    estimator_model()
