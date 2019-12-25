from keras_preprocessing.text import Tokenizer
import numpy as np
import logging
import os
from utils import build_matrix, load_embedding, load_all_texts

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 参数
EMBEDDING_FILE = './data/sgns.renmin.bigram-char'
data_base = './data'
train_file = os.path.join(data_base, 'train.txt')
dev_file = os.path.join(data_base, 'dev.txt')
vocab_file = os.path.join(data_base, 'vocab.txt')
clipped_file = os.path.join(data_base, 'sgn_renmin')  # 持久化embedding矩阵的文件名

# 加载所有的数据并构建词表
all_texts = load_all_texts([train_file, dev_file])
tokenizer = Tokenizer(num_words=None, lower=False)
tokenizer.fit_on_texts(all_texts)
logger.info("the size of vocabulary is {}".format(len(tokenizer.word_counts)))

# 加载词向量
embeddings_index = load_embedding(EMBEDDING_FILE)
logger.info("the size of embedding is {}".format(len(embeddings_index)))  # 预训练词向量的数量

# 按词表裁剪词向量并构建矩阵
glove_embedding_matrix = build_matrix(embeddings_index, tokenizer.word_index)
logger.info("the shape of embedding matrix is {}".format(glove_embedding_matrix.shape))

# 持久化
np.save(clipped_file, glove_embedding_matrix)  # 持久化embedding矩阵
words = [word+'\n' for word in list(tokenizer.word_index.keys())]  # 持久化词表
with open(vocab_file, 'w', encoding='utf-8') as f:
    f.writelines(words)

