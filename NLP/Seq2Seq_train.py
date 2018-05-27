# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import collections
import codecs
from operator import itemgetter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
config.gpu_options.allocator_type = 'BFC'

MAX_LEN = 50  # 限定句子的最大单词数量
SOS_ID = 1  # 目标语言词汇表中<sos>的ID

# 源语言输入文件
SRC_TRAIN_DATA = r'D:\codes\python\tensorflow\book\NLP\translation.en'
# 目标语言输入文件
TRG_TRAIN_DATA = r'D:\codes\python\tensorflow\book\NLP\translation.zh'
# checkpoint保存路径
CHECKPOINT_PATH = (r'D:\codes\python\tensorflow\book\NLP'
                   r'\seq2seq_checkpoint\seq2seq_ckpt')

HIDDEN_SIZE = 1024  # LSTM的隐藏层规模
NUM_LAYERS = 2  # 深层循环神经网络中LSTM结构的层数
SRC_VOCAB_SIZE = 10000  # 源语言词汇表大小
TRG_VOCAB_SIZE = 4000  # 目标语言词汇表大小
BATCH_SIZE = 100  # 训练数据batch的大小
NUM_EPOCH = 5  # 训练轮数
KEEP_PROB = 0.8  # 节点不被dropout的概率
MAX_GRAD_NORM = 5  # 用于控制梯度爆炸的梯度大小上限
SHARE_EMB_AND_SOFTMAX = True  # 在Softmax层和词向量层之间共享参数


def create_vocab():
    """
    将PTB原始输入文件转换成词汇表
    :return:
    """
    ROOT = (r'D:\codes\python\tensorflow\tensorflow-tutorial'
            r'\Deep_Learning_with_TensorFlow\datasets\TED_data')
    RAW_DATA = ROOT + '\\train.txt.zh'
    VOCAB_OUTPUT = (r'D:\codes\python\tensorflow\book'
                    r'\NLP\translation.vocab.zh')

    counter = collections.Counter()
    with codecs.open(RAW_DATA, 'r', 'utf-8') as f:
        for line in f:
            for word in line.strip().split():
                counter[word] += 1

    sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1),
                                reverse=True)
    sorted_words = [x[0] for x in sorted_word_to_cnt]

    # 加入句尾结束符<eos>到sorted_words列表的头部
    sorted_words = ['<unk>', '<sos>', '<eos>'] + sorted_words
    # 英文en是10000，中文zh是4000，如果设置不对运行时loss的值会变成nan
    if len(sorted_words) > 4000:
        sorted_words = sorted_words[:4000]

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
            r'\Deep_Learning_with_TensorFlow\datasets\TED_data')
    RAW_DATA = ROOT + '\\train.txt.zh'
    OUTPUT_DATA = r'D:\codes\python\tensorflow\book\NLP\translation.zh'
    VOCAB = r'D:\codes\python\tensorflow\book\NLP\translation.vocab.zh'

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


def make_dataset(file_path):
    """
    使用Dataset从一个文件中读取一个语言的数据
    数据的格式为每行一句话，单词已经转化为单词编号。也就是说输入的文件是已经将
    单词转化为ID的文件
    :param file_path: 输入文件路径名
    :return: Dataset
    """
    dataset = tf.data.TextLineDataset(file_path)

    # 根据空格将单词编号且分开并放入一个一维向量
    dataset = dataset.map(lambda string: tf.string_split([string]).values)

    # 将字符串形式的单词编号转化为整数
    dataset = dataset.map(lambda string: tf.string_to_number(string, tf.int32))

    # 统计每个句子的单词数量，并与句子内容一起放入Dataset中
    dataset = dataset.map(lambda x: (x, tf.size(x)))

    return dataset


def make_src_trg_dataset(src_path, trg_path, batch_size):
    """
    从源语言文件src_path和目标语言文件trg_path中分别读取数据，并进行填充
    和batching操作
    :param src_path: 源语言文件
    :param trg_path: 目标语言文件
    :param batch_size: batch的大小
    :return: Dataset
    """
    # 首先分别读取源语言数据和目标语言数据
    src_data = make_dataset(src_path)
    trg_data = make_dataset(trg_path)

    # 通过zip操作将两个Dataset合并为一个Dataset。现在每个Dataset中每一项数据ds
    # 由4个Tensor组成：
    # ds[0][0]是源句子
    # ds[0][1]是源句子长度
    # ds[1][0]是目标句子
    # ds[1][1]是目标句子长度
    dataset = tf.data.Dataset.zip((src_data, trg_data))

    def filter_length(src_tuple, trg_tuple):
        """
        删除内容为空（只包含<eos>）的句子和长度过长的句子
        :param src_tuple:
        :param trg_tuple:
        :return:
        """
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        src_len_ok = tf.logical_and(tf.greater(src_len, 1),
                                    tf.less_equal(src_len, MAX_LEN))
        trg_len_ok = tf.logical_and(tf.greater(trg_len, 1),
                                    tf.less_equal(trg_len, MAX_LEN))
        return tf.logical_and(src_len_ok, trg_len_ok)

    dataset = dataset.filter(filter_length)

    def make_trg_input(src_tuple, trg_tuple):
        """
        解码器的输入(trg_input)形式为："<sos> a b c"
        解码器的目标输出(trg_label)形式为："a b c <eos>"
        而经过filter_length()之后得到的目标句子是"a b c <eos>"形式，我们需要从
        中生成"<sos> a b c"形式并加入到Dataset中
        :param src_tuple:
        :param trg_tuple:
        :return:
        """
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)

        # trg_label最后一个id为<eos>，所以截断。再把<sos>加到trg_input开头
        trg_input = tf.concat([[SOS_ID], trg_label[: -1]], axis=0)
        return (src_input, src_len), (trg_input, trg_label, trg_len)

    dataset = dataset.map(make_trg_input)

    # 随机打乱训练数据
    dataset = dataset.shuffle(10000)

    # 规定填充后输出的数据维度
    padded_shapes = (
        (tf.TensorShape([None]),    # 源句子是长度未知的向量
         tf.TensorShape([])),       # 源句子长度是单个数字
        (tf.TensorShape([None]),    # 目标句子(解码器输入)是长度未知的向量
         tf.TensorShape([None]),    # 目标句子(解码器目标输出)是长度未知的向量
         tf.TensorShape([])))       # 目标句子长度是单个数字

    # 调用padded_batch方法进行batching操作
    batched_dataset = dataset.padded_batch(batch_size, padded_shapes)
    return batched_dataset


# 定义NMTModel类来描述模型
class NMTModel(object):
    # 在模型的初始化函数中定义模型要用到的变量
    def __init__(self):
        # 定义编码器和解码器所使用的LSTM结构
        self.enc_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in
             range(NUM_LAYERS)])
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in
             range(NUM_LAYERS)])

        # 为源语言和目标语言分别定义词向量
        self.src_embedding = tf.get_variable('src_emb',
                                             [SRC_VOCAB_SIZE, HIDDEN_SIZE])
        self.trg_embedding = tf.get_variable('trg_emb',
                                             [TRG_VOCAB_SIZE, HIDDEN_SIZE])

        # 定义Softmax层的变量
        if SHARE_EMB_AND_SOFTMAX:
            self.softmax_weight = tf.transpose(self.trg_embedding)
        else:
            self.softmax_weight = tf.get_variable('weight',
                                                  [HIDDEN_SIZE, TRG_VOCAB_SIZE])
        self.softmax_bias = tf.get_variable('bias', [TRG_VOCAB_SIZE])

    def forward(self, src_input, src_size, trg_input, trg_label, trg_size):
        """
        在forward函数中定义模型的前向计算图。5个参数分别是make_src_trg_dataset()
        的返回值
        :param src_input:
        :param src_size:
        :param trg_input:
        :param trg_label:
        :param trg_size:
        :return:
        """
        batch_size = tf.shape(src_input)[0]

        # 将输入和输出单词编号转为词向量
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)
        trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)

        # 在词向量上进行dropout
        src_emb = tf.nn.dropout(src_emb, KEEP_PROB)
        trg_emb = tf.nn.dropout(trg_emb, KEEP_PROB)

        # 使用dynamic_rnn构造编码器
        # 编码器读取源句子每个位置的词向量，输出最后一步的隐藏状态enc_state
        # 因为编码器是一个双层LSTM，因此enc_state是一个包含两个LSTMStateTuple类
        # 的tuple，每个LSTMStateTuple对应编码器中一层的状态。
        # enc_outputs是顶层LSTM在每一步的输出，它的维度是
        # [batch_size, max_time, HIDDEN_SIZE]
        # Seq2Seq模型中不需要用到enc_outputs
        with tf.variable_scope('encoder'):
            enc_outputs, enc_state = tf.nn.dynamic_rnn(self.enc_cell, src_emb,
                                                       src_size,
                                                       dtype=tf.float32)

        # 使用dynamic_rnn构造解码器
        # 解码器读取源句子每个位置的词向量，输出的dec_outputs为每一步
        # 顶层LSTM的输出。dec_outputs的维度是
        # [batch_size, max_time, HIDDEN_SIZE]
        # initial_state=enc_state 表示用编码器的输出来初始化第一步的隐藏状态
        with tf.variable_scope('decoder'):
            dec_outputs, _ = tf.nn.dynamic_rnn(self.dec_cell, trg_emb, trg_size,
                                               initial_state=enc_state)

        # 计算解码器每一步的log perplexity
        output = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
        logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(trg_label, [-1]), logits=logits)

        # 在计算平均损失时，需要将填充位置的权重置为0，以避免无效位置的预测
        # 干扰模型的训练
        label_weights = tf.sequence_mask(trg_size,
                                         maxlen=tf.shape(trg_label)[1],
                                         dtype=tf.float32)
        label_weights = tf.reshape(label_weights, [-1])
        cost = tf.reduce_sum(loss * label_weights)
        cost_per_token = cost / tf.reduce_sum(label_weights)

        # 定义反向传播操作
        trainable_variables = tf.trainable_variables()

        # 控制梯度大小，定义优化方法和训练步骤
        grads = tf.gradients(cost / tf.to_float(batch_size),
                             trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_op = optimizer.apply_gradients(zip(grads, trainable_variables))
        return cost_per_token, train_op


def run_epoch(session, cost_op, train_op, saver, step):
    """
    使用给定的模型model训练一个epoch
    每训练200步保存一个checkpoint
    :param session:
    :param cost_op:
    :param train_op:
    :param saver:
    :param step:
    :return:
    """
    # 训练一个epoch
    # 重复训练步骤直至遍历完Dataset中所有数据
    while True:
        try:
            # 运行train_op并计算损失值。训练数据在main()函数中以Dataset方式提供
            cost, _ = session.run([cost_op, train_op])
            # print("After %d steps, per token cost is %.3f" % (step, cost))
            if step % 10 == 0:
                print("After %d steps, per token cost is %.3f" %
                      (step, cost))

            # 每200步保存一个checkpoint
            if step % 200 == 0:
                saver.save(session, CHECKPOINT_PATH, global_step=step)

            step += 1
        except tf.errors.OutOfRangeError:
            break
    return step


def main():
    # 定义初始化函数
    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    # 定义训练用的循环神经网络模型
    with tf.variable_scope('nmt_model', reuse=None, initializer=initializer):
        train_model = NMTModel()

    # 定义输入数据
    data = make_src_trg_dataset(SRC_TRAIN_DATA, TRG_TRAIN_DATA, BATCH_SIZE)
    iterator = data.make_initializable_iterator()
    (src, src_size), (trg_input, trg_label, trg_size) = iterator.get_next()

    # 定义前向计算图。输入数据以张量形式提供给forward函数
    cost_op, train_op = train_model.forward(src, src_size,
                                            trg_input, trg_label, trg_size)

    # 训练模型
    saver = tf.train.Saver()
    step = 0
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        for i in range(NUM_EPOCH):
            print("In iteration: %d" % (i + 1))
            sess.run(iterator.initializer)
            step = run_epoch(sess, cost_op, train_op, saver, step)


if __name__ == '__main__':
    # create_vocab()
    # vocab_to_id()
    main()
