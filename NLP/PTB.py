import os
import numpy as np
import tensorflow as tf
import codecs
import collections
from operator import itemgetter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allocator_type = 'BFC'

# 处理PTB时用到的一些参数
TRAIN_DATA = r'D:\codes\python\tensorflow\book\NLP\ptb.train'  # 训练数据
EVAL_DATA = r'D:\codes\python\tensorflow\book\NLP\ptb.valid'  # 验证数据
TEST_DATA = r'D:\codes\python\tensorflow\book\NLP\ptb.test'  # 测试数据
HIDDEN_SIZE = 300  # RNN中隐藏层的层数
NUM_LAYERS = 2  # LSTM结构的层数
VOCAB_SIZE = 10000  # 词典规模
TRAIN_BATCH_SIZE = 20  # 一个batch的行数
TRAIN_NUM_STEP = 35  # 一个batch的列数（训练数据截断长度）

EVAL_BATCH_SIZE = 1  # 测试数据batch大小（行数）
EVAL_NUM_STEP = 1  # 一个测试数据batch的列数（测试数据截断长度）
NUM_EPOCH = 5  # 使用训练数据的轮数
LSTM_KEEP_PROB = 0.9  # LSTM节点不被dropout的概率
EMBEDDING_KEEP_PROB = 0.9  # 词向量不被dropout的概率
MAX_GRAD_NORM = 5  # 用于控制梯度膨胀的梯度大小上限
SHARE_EMB_AND_SOFTMAX = True  # 再Softmax层和词向量层之间共享参数


def create_vocab():
    """
    将PTB原始输入文件转换成词汇表
    :return:
    """
    ROOT = (r'D:\codes\python\tensorflow\tensorflow-tutorial'
            r'\Deep_Learning_with_TensorFlow\datasets\PTB_data')
    RAW_DATA = ROOT + '\ptb.train.txt'
    VOCAB_OUTPUT = r'D:\codes\python\tensorflow\book\NLP\ptb.vocab'

    counter = collections.Counter()
    with codecs.open(RAW_DATA, 'r', 'utf-8') as f:
        for line in f:
            for word in line.strip().split():
                counter[word] += 1

    sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1),
                                reverse=True)
    sorted_words = [x[0] for x in sorted_word_to_cnt]

    # 加入句尾结束符<eos>到sorted_words列表的头部
    sorted_words = ['<eos>'] + sorted_words

    with codecs.open(VOCAB_OUTPUT, 'w', 'utf-8') as file_output:
        for word in sorted_words:
            file_output.write(word + '\n')


def vocab_to_id():
    """
    将词汇表中的单词转换成对应的id，得到训练文件中单词对应的id的列表。
    也就是将训练文件中的句子转换成了id列表
    :return:
    """
    ROOT = (r'D:\codes\python\tensorflow\tensorflow-tutorial'
            r'\Deep_Learning_with_TensorFlow\datasets\PTB_data')
    RAW_DATA = ROOT + '\ptb.test.txt'
    OUTPUT_DATA = r'D:\codes\python\tensorflow\book\NLP\ptb.test'
    VOCAB = r'D:\codes\python\tensorflow\book\NLP\ptb.vocab'

    with codecs.open(VOCAB, 'r', 'utf-8') as f_vocab:
        vocab = [w.strip() for w in f_vocab.readlines()]

    word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}

    def get_id(word):
        return word_to_id[word] if word in word_to_id else word_to_id['<unk>']

    fin = codecs.open(RAW_DATA, 'r', 'utf-8')
    fout = codecs.open(OUTPUT_DATA, 'w', 'utf-8')

    for line in fin:
        words = line.strip().split() + ['<eos>']
        out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
        fout.write(out_line)

    fin.close()
    fout.close()


def read_data(file_path):
    """
    获取训练文件中句子对应的id列表
    :param file_path: 用id替换单词之后的文件
    :return:
    """
    with open(file_path, 'r') as fin:
        id_string = ' '.join([line.strip() for line in fin.readlines()])
    id_list = [int(w) for w in id_string.split()]
    return id_list


def make_batches(id_list, batch_size, num_step):
    # 计算需要分成多少个batch，减一是因为在词汇表中加入了句尾结束符<eos>
    # 一个batch的大小是 [batch_size, num_step]
    num_batches = (len(id_list) - 1) // (batch_size * num_step)

    # 取出能组成num_batches个batch的元素进行batch划分
    data = np.array(id_list[: num_batches * batch_size * num_step])

    # 先把取出的元素划分成batch_size行
    data = np.reshape(data, [batch_size, num_step * num_batches])

    # 再对data的列进行划分，也就是第二维(axis=1)，这样得到的batch的大小就是
    # [batch_size, num_step]
    data_batches = np.split(data, num_batches, axis=1)

    # 重复上述划分操作，为特征batch准备对应的label batch。由于是用前n个预测
    # 第n+1个id，所以需要右移一位
    label = np.array(id_list[1: num_batches * batch_size * num_step + 1])
    label = np.reshape(label, [batch_size, num_step * num_batches])
    label_batches = np.split(label, num_batches, axis=1)

    # 返回一个长度为num_batches的数组，其中每一项为一个元组，是data矩阵和
    # label矩阵的组合
    return list(zip(data_batches, label_batches))


class PTBModel(object):
    """
    通过一个PTBModel类来描述语言模型，这样方便维护循环神经网络中的状态
    """
    def __init__(self, is_training, batch_size, num_steps):
        # 记录使用的batch大小和截断长度
        self.batch_size = batch_size
        self.num_steps = num_steps

        # 定义每一步的输入和预期输出。输入输出的维度都是[batch_size, num_steps]
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        # 定义使用LSTM结构为循环体结构且使用dropout的深层循环神经网络
        dropout_keep_prob = LSTM_KEEP_PROB if is_training else 1.0
        lstm_cells = [
            tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE),
                output_keep_prob=dropout_keep_prob)
            for _ in range(NUM_LAYERS)]
        cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

        # 初始化最初的状态，即全零的向量。这个量只在每个epoch初始化第一个batch
        # 时使用
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        # 定义单词的词向量矩阵
        embedding = tf.get_variable('embedding', [VOCAB_SIZE, HIDDEN_SIZE])

        # 将输入单词转化为词向量
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        # 只在训练时使用dropout
        if is_training:
            inputs = tf.nn.dropout(inputs, EMBEDDING_KEEP_PROB)

        # 定义输出列表。在这里先将不同时刻LSTM结构的输出收集起来，再一起提供给
        # softmax层
        outputs = []
        state = self.initial_state
        with tf.variable_scope('RNN'):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                # 这里cell_output的大小就是[batch, hidden_size]
                cell_output, state = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        # 把输出队列展开成[batch, hidden_size * num_steps]的形状，然后再
        # reshape成[batch * num_steps, hidden_size]的形状
        # tf.concat()返回的结果就是行不变，在列上叠加了num_steps次
        # [-1, hidden_size]中的-1表示由程序自动计算，一个shape列表中只能出现一次
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])

        # Softmax层：将RNN在每个位置上的输出转化为各个单词的logits
        if SHARE_EMB_AND_SOFTMAX:
            weight = tf.transpose(embedding)
        else:
            weight = tf.get_variable('weight', [HIDDEN_SIZE, VOCAB_SIZE])
        bias = tf.get_variable('bias', [VOCAB_SIZE])
        logits = tf.matmul(output, weight) + bias

        # 定义交叉熵损失函数和平均损失
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.targets, [-1]), logits=logits)
        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state

        # 只在训练模型时定义反向传播操作
        if not is_training:
            return

        trainable_variables = tf.trainable_variables()

        # 控制梯度大小，定义优化方法和训练步骤
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.cost, trainable_variables), MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        self.train_op = optimizer.apply_gradients(
            zip(grads, trainable_variables))


def run_epoch(session, model, batches, train_op, output_log, step):
    """
    使用给定的模型model在数据batches上运行train_op并返回全部数据上的perplexity
    :param session:
    :param model:
    :param batches:
    :param train_op:
    :param output_log: 只有在训练时为True，输出日志
    :param step:
    :return:
    """
    # 计算平均perplexity的辅助变量
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    # 训练一个epoch
    for x, y in batches:
        # 在当前batch上运行train_op并计算损失值。交叉熵损失函数计算的就是下一个
        # 单词为给定单词的概率
        cost, state, _ = session.run([model.cost, model.final_state, train_op],
                                     {model.input_data: x, model.targets: y,
                                      model.initial_state: state})
        total_costs += cost
        iters += model.num_steps

        # 只有在训练时输出日志
        if output_log and step % 100 == 0:
            print("After %d steps, perplexity is %.3f" % (
                step, np.exp(total_costs / iters)))
        step += 1

    # 返回给定模型在给定数据上的perplexity值
    return step, np.exp(total_costs / iters)


def main():
    # 定义初始化函数
    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    # 定义训练用的循环神经网络模型
    with tf.variable_scope('language_model',
                           reuse=None, initializer=initializer):
        train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)

    # 定义测试用的循环神经网络模型。它与train_model共用参数，但是没有dropout
    with tf.variable_scope('language_model',
                           reuse=True, initializer=initializer):
        eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)

    # 训练模型
    with tf.Session(config=config) as session:
        tf.global_variables_initializer().run()
        train_batches = make_batches(read_data(TRAIN_DATA), TRAIN_BATCH_SIZE,
                                     TRAIN_NUM_STEP)
        eval_batches = make_batches(read_data(EVAL_DATA), EVAL_BATCH_SIZE,
                                    EVAL_NUM_STEP)
        test_batches = make_batches(read_data(TEST_DATA), EVAL_BATCH_SIZE,
                                    EVAL_NUM_STEP)

        step = 0
        for i in range(NUM_EPOCH):
            print("In iteration: %d" % (i + 1))
            step, train_pplx = run_epoch(session, train_model, train_batches,
                                         train_model.train_op, True, step)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_pplx))

            _, eval_pplx = run_epoch(session, eval_model, eval_batches,
                                     tf.no_op(), False, 0)
            print("Epoch: %d Eval Perplexity: %.3f" % (i + 1, eval_pplx))

        _, test_pplx = run_epoch(session, eval_model, test_batches, tf.no_op(),
                                 False, 0)
        print("Test Perplexity: %.3f" % test_pplx)


if __name__ == '__main__':
    main()
