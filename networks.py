import tensorflow as tf
import numpy as np
from tensorflow.contrib.lookup import index_table_from_file


class BiLSTMCRF(object):
    def __init__(self, inp, labels, mode, params):
        self.params = params  # 保存参数
        self.mode = mode
        self.training = (mode == tf.estimator.ModeKeys.TRAIN)
        self.num_tags = len(params['tags'])

        words, nwords = inp
        vocab_words = index_table_from_file(params['vocab'],
                                            num_oov_buckets=params['num_oov_buckets'])  # 从文件中构造词表与id的映射
        # 将词转换为id，对于袋外词会转换为当前最大id加1. 如词表中最大的id为10,那么所有袋外词的id均为11
        word_ids = vocab_words.lookup(words)

        with tf.variable_scope("embedding"):
            if params['use_pretrained']:  # 是否使用预训练词向量
                glove = np.load(params['embed'])
                glove = np.vstack([glove, [[0.] * params['dim']]])  # 将全0向量拼接到glove矩阵的最下边，作为袋外词的向量
                W = tf.Variable(glove, dtype=tf.float32, trainable=True)
            else:
                W = tf.Variable(tf.random_uniform([params['vocab_size'], params['dim']], -1.0, 1.0),
                                name='W',
                                trainable=True)
            embeddings = tf.nn.embedding_lookup(W, word_ids)
            # (batch_size,seq_len,embedding_dim)
            encoder_input = tf.layers.dropout(embeddings, rate=params['dropout'], training=self.training)

        encoder_outputs = [encoder_input]  # 保存多层LSTM的输出结果
        for i in range(params['lstm_layers']):
            with tf.variable_scope("BiLSTM"+str(i)):
                encoder_output = self.BiLSTM(encoder_outputs[-1], nwords)
                encoder_outputs.append(encoder_output)

        with tf.variable_scope("CRF"):
            logits, self.pred_ids, crf_params, self.score = self.CRF(encoder_outputs[-1], self.num_tags, nwords)

        reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_tensor(params['tags'])  # 反向词表
        self.pred_strings = reverse_vocab_tags.lookup(tf.to_int64(self.pred_ids))  # 将预测的id转换为对应的tag

        with tf.variable_scope("loss"):
            if self.mode != tf.estimator.ModeKeys.PREDICT:
                vocab_tags = tf.contrib.lookup.index_table_from_tensor(params['tags'])  # tags的词表
                self.tags = vocab_tags.lookup(labels)  # 将tags转换为对应的id
                log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                    logits, self.tags, nwords, crf_params)
                self.loss = tf.reduce_mean(-log_likelihood)
                self.weights = tf.sequence_mask(nwords)
                self.train_op = tf.train.AdamOptimizer().minimize(
                    self.loss, global_step=tf.train.get_or_create_global_step())

    def BiLSTM(self, inputs, seq_length):
        t = tf.transpose(inputs, perm=[1, 0, 2])  # (seq_len,batch_size,embedding_dim)
        lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(self.params['lstm_size'])
        lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(self.params['lstm_size'])
        lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
        # (seq_len,batch_size,lstm_size)
        output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=seq_length)
        output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=seq_length)
        # (seq_len,batch_size,lstm_size*2)
        output = tf.concat([output_fw, output_bw], axis=-1)
        # (batch_size,seq_len,lstm_size*2)
        output = tf.transpose(output, perm=[1, 0, 2])
        output = tf.layers.dropout(output, rate=self.params['dropout'], training=self.training)
        return output

    def CRF(self, inputs, num_tags, seq_length):
        logits = tf.layers.dense(inputs, num_tags)
        crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)
        # (batch_size,seq_len)
        pred_ids, score = tf.contrib.crf.crf_decode(logits, crf_params, seq_length)
        return logits, pred_ids, crf_params, score
