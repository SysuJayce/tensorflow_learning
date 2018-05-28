import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

# disable AVX warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)


def example():
    mnist = input_data.read_data_sets(
        r'D:\codes\python\tensorflow\tensorflow_learning\datasets\MNIST_data',
        one_hot=True)

    print('training data size: %d' % mnist.train.num_examples)
    print('validating date size: %d' % mnist.validation.num_examples)
    print('testing data size: %d' % mnist.test.num_examples)

    print('example training data: ', mnist.train.images[0])
    print('example training data label: ', mnist.train.labels[0])


def mnist_train(mnist):
    # num of input_layer node and output_layer node
    INPUT_NODE = 784
    OUTPUT_NODE = 10

    # num of hidden_layer node
    LAYER1_NODE = 500

    BATCH_SIZE = 100
    LEARNING_RATE_BASE = 0.8
    LEARNING_RATE_DECAY = 0.99
    REGULARIZATION_RATE = 0.0001
    TRAINING_STEPS = 30000
    MOVING_AVERAGE_DECAY = 0.99

    '''
    augxiliary function to get forward propagation result(predict result for
    training data)
    '''
    def inference(input_tensor, avg_class, weight1, bias1, weight2, bias2):
        if avg_class is None:
            layer1 = tf.nn.relu(tf.matmul(input_tensor, weight1) + bias1)
            '''
            since softmax will be called when calculate loss_function,
            no need to apply activation_function now.  
            Besides, softmax won't change the predict result because
            the result is a 1-dimension vector, we take the index of
            the max value(1), and softmax only change the result value
            to probability, so no need to apply softmax now.
           '''
            return tf.matmul(layer1, weight2) + bias2
        else:
            # if avg_class is provided, use it to calculate a moving average
            # value of weights and bias
            layer1 = tf.nn.relu(
                tf.matmul(input_tensor, avg_class.average(weight1)) +
                avg_class.average(bias1))
            return (tf.matmul(layer1, avg_class.average(weight2)) +
                    avg_class.average(bias2))

    def train(mnist):
        x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

        # generate parameter for hidden_layer
        weight1 = tf.Variable(
            tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
        bias1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

        # generate parameter for output_layer
        weight2 = tf.Variable(
            tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
        bias2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

        # get forward propagation result y without moving average class
        y = inference(x, None, weight1, bias1, weight2, bias2)

        # set a global_step to record total trained times, no need to train it
        global_step = tf.Variable(0, trainable=False)

        # initiate moving average class
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)

        # apply moving average to trainable variable
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

        # get forward propagation result y with moving average class
        average_y = inference(
            x, variable_averages, weight1, bias1, weight2, bias2)

        # get cross_entropy
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y, labels=tf.argmax(y_, 1))

        # get average cross_entropy of current batch
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        # get L2 regularization loss function
        regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

        # get total regularization loss of weights without biases
        regularization = regularizer(weight1) + regularizer(weight2)

        # total loss is the sum of cross_entropy loss and regularization loss
        loss = cross_entropy_mean + regularization

        # set decay learning rate
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE, global_step,
            mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)

        # use gradient_decent to optimize loss function
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
            loss, global_step)

        # update parameters of neural network and their's moving average value
        # after back propagation together
        with tf.control_dependencies([train_step, variable_averages_op]):
            train_op = tf.no_op(name='train')

        # get the accuracy
        correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            tf.global_variables_initializer().run()

            # 用于验证的数据
            validate_feed = {x: mnist.validation.images,
                             y_: mnist.validation.labels}

            # 用于测试的数据
            test_feed = {x: mnist.test.images,
                         y_: mnist.test.labels}

            # 迭代训练神经网络
            for i in range(TRAINING_STEPS):
                # 每1000轮输出一次再验证数据集上的测试结果
                if i % 1000 == 0:
                    # 当i为0时，placeholder的第一维None为0，也可以执行
                    validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %d training step(s), validation accuracy"
                          " using average model is %g" % (i, validate_acc))

                # 产生这一轮的batch个训练数据
                xs, ys = mnist.train.next_batch(BATCH_SIZE)
                sess.run(train_op, feed_dict={x: xs, y_: ys})

            # 训练结束后在测试数据上计算最终正确率
            test_acc = sess.run(accuracy, feed_dict=test_feed)
            print("After %d training step(s), test accuracy"
                  " using average model is %g" % (TRAINING_STEPS, test_acc))

    train(mnist)


def main(argv=None):
    # 声明处理MNIST数据集的类，这个类在初始化时会自动下载数据
    mnist = input_data.read_data_sets(
        r'D:\codes\python\tensorflow\tensorflow_learning\datasets\MNIST_data',
        one_hot=True)
    mnist_train(mnist)


def test():
    v = tf.get_variable("v", shape=[1], initializer=tf.constant_initializer(1.0))
    print(v.name)
    v = tf.Variable(tf.constant(1.0, shape=[1]), name="v")
    print(v.name)

    with tf.variable_scope("foo", reuse=False):
        v = tf.get_variable("v", shape=[1], initializer=tf.ones_initializer)

    with tf.variable_scope("foo", reuse=True):
        v = tf.get_variable("v", shape=[1])


def persistent_save():
    from tensorflow.python.framework import graph_util

    v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
    v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
    result = v1 + v2

    init_op = tf.global_variables_initializer()

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init_op)

        graph_def = tf.get_default_graph().as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, graph_def, ['add']
        )
        with tf.gfile.GFile(r"D:\codes\python\tensorflow\book\persistent\model.ckpt",
                            "wb") as f:
            f.write(output_graph_def.SerializeToString())

    # v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
    # v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
    # result = v1 + v2
    #
    # init_op = tf.global_variables_initializer()
    #
    # saver = tf.train.Saver()
    #
    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    #     sess.run(init_op)
    #     saver.save(sess, r"D:\codes\python\tensorflow\book\persistent\model.ckpt")

    # v = tf.Variable(0, dtype=tf.float32, name="v")
    # for variables in tf.global_variables():
    #     print(variables.name)
    #
    # ema = tf.train.ExponentialMovingAverage(0.99)
    # maintain_average_op = ema.apply(tf.global_variables())
    # for variables in tf.global_variables():
    #     print(variables.name)
    #
    # saver = tf.train.Saver()
    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    #     init_op = tf.global_variables_initializer()
    #     sess.run(init_op)
    #
    #     sess.run(tf.assign(v, 10))
    #     sess.run(maintain_average_op)
    #     saver.save(sess, r"D:\codes\python\tensorflow\book\persistent\model.ckpt")
    #     print(sess.run([v, ema.average(v)]))


def persistent_restore():
    v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
    v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
    result = v1 + v2

    saver = tf.trdain.Saver()
    saver.export_meta_graph(r"D:\codes\python\tensorflow\book\persistent\model.ckpt.meta.json",
                            as_text=True)
    # from tensorflow.python.platform import gfile
    #
    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    #     model_filename = r"D:\codes\python\tensorflow\book\persistent\model.ckpt"
    #     with gfile.FastGFile(model_filename, 'rb') as f:
    #         graph_def = tf.GraphDef()
    #         graph_def.ParseFromString(f.read())
    #
    # result = tf.import_graph_def(graph_def, return_elements=["add:0"])
    # print(sess.run(result))

    # v = tf.Variable(0, dtype=tf.float32, name="v")
    # ema = tf.train.ExponentialMovingAverage(0.99)
    #
    # print(ema.variables_to_restore())
    #
    # saver = tf.train.Saver(ema.variables_to_restore())
    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    #     saver.restore(sess, r"D:\codes\python\tensorflow\book\persistent\model.ckpt")
    #     print(sess.run(v))

    # v = tf.Variable(0, dtype=tf.float32, name="v")
    # saver = tf.train.Saver({"v/ExponentialMovingAverage": v})
    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    #     saver.restore(sess, r"D:\codes\python\tensorflow\book\persistent\model.ckpt")
    #     print(sess.run(v))

    # v1 = tf.get_variable("v1", [1], initializer=tf.constant_initializer(1.0))
    # v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v22")
    # result = v1 + v2
    #
    # saver = tf.train.Saver({"v1":v1, "v2":v2})
    #
    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    #     saver.restore(sess, r"D:\codes\python\tensorflow\book\persistent\model.ckpt")
    #     print(sess.run(result))

    # saver = tf.train.import_meta_graph(
    #     r"D:\codes\python\tensorflow\book\persistent\model.ckpt.meta")
    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    #     saver.restore(sess, r"D:\codes\python\tensorflow\book\persistent\model.ckpt")
    #     print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))


if __name__ == '__main__':
    # example()
    # 查找并执行程序中定义的main函数
    # tf.app.run()
    # mnist_train()
    # test()
    persistent_restore()
    # persistent_save()