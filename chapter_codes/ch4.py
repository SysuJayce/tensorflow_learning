import tensorflow as tf
import os

# Disable AVX warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)


def test():
    v1 = tf.constant([1.0, 2.0, 3.0, 4.0])
    v2 = tf.constant([4.0, 3.0, 2.0, 1.0])

    weights = tf.constant([[1.0, -2.0], [-3.0, 4.0]])

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        print(sess.run(tf.contrib.layers.l1_regularizer(.5)(weights)))
        print(sess.run(tf.contrib.layers.l2_regularizer(.5)(weights)))
        # print(tf.greater(v1, v2).eval())
        # print(tf.where(tf.greater(v1, v2), v1, v2).eval())
        # print((v1 * v2).eval())
        # print(tf.matmul(v1, v2).eval())

    # v1 = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    #
    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    #     print(tf.reduce_mean(v).eval())


def loss_func():
    from numpy.random import RandomState

    batch_size = 8

    # 两个输入节点
    x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
    # 回归问题一般只有一个输出节点
    y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

    # 定义一个单层的神经网络前向传播的过程，这里就是简单的加权和
    w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
    y = tf.matmul(x, w1)

    # 定义预测过多和过少的成本
    loss_less = 10
    loss_more = 1
    loss = tf.reduce_sum(tf.where(tf.greater(y, y_),
                                  (y - y_) * loss_more, (y_ - y) * loss_less))
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

    # 通过随机数生成一个模拟数据集
    rdm = RandomState(1)
    dataset_size = 128
    X = rdm.rand(dataset_size, 2)

    '''
    设置数据集的真实值为输入的和加上一个随机量。之所以加上随机量是为了
    加入不可预测的噪音，否则不同损失函数的意义就不大了，因为不同损失函数都
    会在能完全预测正确的时候最低。
    一般来说噪音为一个均值为0的小量，这里噪音设置为[-0.05, 0.05]的随机数
    '''
    Y = [[x1 + x2 + rdm.rand() / 10.0 - 0.05] for (x1, x2) in X]
    # Y = [[x1 + x2] for (x1, x2) in X]

    # 训练神经网络
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        steps = 5000
        for i in range(steps):
            start = (i * batch_size) % dataset_size
            end = min(start + batch_size, dataset_size)
            sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})

        print(sess.run(w1))


def regularization():
    # 获取一层神经网络边上的权重，并将这个权重的L2正则化损失加入名称为
    # 'losses'的集合中
    def get_weight(shape, lambda_):
        var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
        # 'add_to_collection'函数将L2正则化损失项加入集合。第一个参数为集合的
        # 名字，第二个参数为要加入集合的内容
        tf.add_to_collection('losses',
                             tf.contrib.layers.l2_regularizer(lambda_)(var))
        return var

    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))
    batch_size = 8

    # 定义每一层网络的节点个数，包括输出层、输出层和3个隐藏层
    layer_dimension = [2, 10, 10, 10, 1]

    # 神经网络的层数
    n_layers = len(layer_dimension)

    # cur_layer变量维护前向传播时最深层的节点(最后会维护到输出层)，开始是输入层
    cur_layer = x

    # 当前层的节点个数(可以理解为这一层的神经元的输入来源数)
    in_dimension = layer_dimension[0]

    # 通过一个循环来生成5层全连接的神经网络结构
    for i in range(1, n_layers):
        # layer_dimension[i]为下一层的节点个数
        out_dimension = layer_dimension[i]
        # 生成当前层中权重的变量，并将其L2正则化损失加入计算图上的集合
        weight = get_weight([in_dimension, out_dimension], 0.001)
        bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
        # 使用ReLU激活函数
        cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
        # 进入下一层之前将下一层的节点个数更新为当前层的节点个数
        in_dimension = layer_dimension[i]

    # 由于前面生成神经网络的时候已经把每一层的L2正则化损失加入了图上的集合，
    # 这里只需要计算损失函数就可以了
    mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))
    tf.add_to_collection('losses', mse_loss)

    # 将losses集合中的元素加起来就是总损失函数
    loss = tf.add_n(tf.get_collection('losses'))


if __name__ == '__main__':
    # loss_func()
    # test()
    regularization()
