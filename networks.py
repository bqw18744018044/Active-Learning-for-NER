import tensorflow as tf
import numpy as np
from tensorflow.contrib.lookup import index_table_from_file


class BaseNetwork(object):
    def __init__(self, inp, labels, mode, params):
        self.inp = inp
        self.labels = labels
        self.params = params
        self.mode = mode
        self.logits = None
        self.pred_ids = None
        self.pred_strings = None
        self.probs = None
        self.score = None
        self.mnlp_score = None
        self.tags = None
        self.num_tags = None
        self.weights = None
        self.loss = None


class BiLstmCrf(BaseNetwork):
    def __init__(self, inp, labels, mode, params):
        self.inp = inp
        self.labels = labels
        self.params = params
        self.mode = mode
        self.training = (mode == tf.estimator.ModeKeys.TRAIN)
        self.num_tags = len(params.tags)
        self.logits = None
        self.pred_ids = None  # 预测结果的id
        self.pred_strings = None  # 预测结果
        self.probs = None  # 使用softmax将logits转换的概率分别
        self.score = None  # 在CRF层中，有viterbi算法生成的结果评分
        self.mnlp_score = None  # 主动学习算法mnlp的评分
        self.tags = None  # 将labels转换为对呀的id
        self.weights = None
        self.loss = None
        self.train_op = None
        self._build()

    def _build(self):
        words, nwords = self.inp  # words：输入的文本, nwords：输入文本的长度
        vocab_words = index_table_from_file(self.params.vocab,
                                            num_oov_buckets=self.params.num_oov_buckets)  # 从文件中构造词表与id的映射
        # 将词转换为id，对于袋外词会转换为当前最大id加1. 如词表中最大的id为10,那么所有袋外词的id均为11
        word_ids = vocab_words.lookup(words)

        with tf.variable_scope("embedding"):
            if self.params.use_pretrained:
                glove = np.load(self.params.embed)
                # 将全0向量拼接到glove矩阵的最下边，作为袋外词的向量
                glove = np.vstack([glove, [[0.] * self.params.embed_dim]])
                W = tf.Variable(glove, dtype=tf.float32, trainable=True)
            else:
                W = tf.Variable(tf.random_uniform([self.params.vocab_size, self.params.embed_dim], -1.0, 1.0),
                                name='W',
                                trainable=True)
            embeddings = tf.nn.embedding_lookup(W, word_ids)
            # (batch_size,seq_len,embedding_dim)
            embeddings = tf.layers.dropout(embeddings, rate=self.params.dropout, training=self.training)

        outputs = []  # 保存多层BiLSTM的输出结果
        with tf.variable_scope("BiLSTM"):
            outputs.append(self.BiLSTM(embeddings, nwords))
            for i in range(self.params.lstm_layer - 1):
                outputs.append(self.BiLSTM(outputs[-1], nwords))

        with tf.variable_scope("CRF"):
            self.logits, self.pred_ids, crf_params, self.score = self.CRF(outputs[-1], self.num_tags, nwords)

        with tf.variable_scope("output"):
            self.probs = tf.nn.softmax(self.logits, axis=-1)
            best_probs = tf.reduce_max(self.probs, axis=-1)
            self.mnlp_score = tf.reduce_mean(tf.log(best_probs), axis=-1)
            reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_tensor(self.params.tags)  # 反向词表
            self.pred_strings = reverse_vocab_tags.lookup(tf.to_int64(self.pred_ids))  # 将预测的id转换为对应的tag
            self.weights = tf.sequence_mask(nwords)

        with tf.variable_scope("loss"):
            if self.mode != tf.estimator.ModeKeys.PREDICT:
                vocab_tags = tf.contrib.lookup.index_table_from_tensor(self.params.tags)  # tags的词表
                self.tags = vocab_tags.lookup(self.labels)  # 将tags转换为对应的id
                log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                    self.logits, self.tags, nwords, crf_params)
                self.loss = tf.reduce_mean(-log_likelihood)
                self.train_op = tf.train.AdamOptimizer().minimize(
                    self.loss, global_step=tf.train.get_or_create_global_step())

    def CRF(self, inputs, num_tags, seq_length):
        logits = tf.layers.dense(inputs, num_tags)
        crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)
        # (batch_size,seq_len)
        pred_ids, score = tf.contrib.crf.crf_decode(logits, crf_params, seq_length)
        return logits, pred_ids, crf_params, score

    def BiLSTM(self, inputs, seq_length):
        t = tf.transpose(inputs, perm=[1, 0, 2])  # (seq_len,batch_size,embedding_dim)
        lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(self.params.lstm_size)
        lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(self.params.lstm_size)
        lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
        # (seq_len,batch_size,lstm_size)
        output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=seq_length)
        output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=seq_length)
        # (seq_len,batch_size,lstm_size*2)
        output = tf.concat([output_fw, output_bw], axis=-1)
        # (batch_size,seq_len,lstm_size*2)
        output = tf.transpose(output, perm=[1, 0, 2])
        output = tf.layers.dropout(output, rate=self.params.dropout, training=self.training)
        return output
