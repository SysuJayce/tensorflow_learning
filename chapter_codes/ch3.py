import os
import tensorflow as tf

# 用于生成模拟数据集
from numpy.random import RandomState

# Disable AVX warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)


def nn():
    # 训练数据batch的大小
    batch_size = 8

    # 参数
    w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
    w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

    # 在placeholder中设置shape的时候不指定行，这样在训练的时候可以输入较小的batch
    # 在测试的时候可以输入较大的batch。在输入数据的时候更自由
    x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
    y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

    # 前向传播求预测值的过程
    a = tf.matmul(x, w1)
    y = tf.matmul(a, w2)

    '''
    后向传播的过程
    '''
    # 将y转换为0-1的数值
    y = tf.sigmoid(y)

    # 定义交叉熵
    cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
                                    + (1 - y) * tf.log(
        tf.clip_by_value(1 - y, 1e-10, 1.0)))
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    # 通过随机数生成模拟数据集
    rdm = RandomState(1)
    dataset_size = 128
    X = rdm.rand(dataset_size, 2)

    # 定义规则来给出样本的标签。这里定义x1+x2<1的样例为正样本，否则为负样本。
    # 在这里标签用0表示负样本，1表示正样本
    Y = [[int(x1 + x2) < 1] for (x1, x2) in X]

    # 创建会话执行TensorFlow程序

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # 初始化变量
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 打印初始参数值
        print("initial weights w1:")
        print(sess.run(w1))
        print("initial weights w2:")
        print(sess.run(w2))

        # 训练参数
        steps = 5000
        for i in range(steps):
            # 每次选取batch个样本进行训练
            start = (i * batch_size) % dataset_size
            end = min(start + batch_size, dataset_size)

            # 通过选取的样本训练神经网络并更新参数
            sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})

            # 每训练1000轮计算一次在所有数据上的交叉熵并打印
            if i % 1000 == 0:
                total_cross_entropy = sess.run(
                    cross_entropy, feed_dict={x: X, y_: Y})
                print("After %d training step(s), cross entropy on all" % i
                      + "data is %g" % total_cross_entropy)

        # 打印训练后的参数值
        print("final weights w1:")
        print(sess.run(w1))
        print("final weights w2:")
        print(sess.run(w2))


if __name__ == '__main__':
    nn()
