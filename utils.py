import codecs
from tqdm import tqdm
import numpy as np


def load_data(file):
    """只读取第1列和最后1列"""
    with codecs.open(file, encoding='utf-8') as f:
        texts = []
        text = []
        labels = []
        label = []
        for line in f:
            line = line.strip()
            if len(line) == 0:  # 空白行，表示一句话已经结束
                texts.append(text)
                labels.append(label)
                text = []
                label = []
            else:
                line = line.split()
                text.append(line[0])
                label.append(line[-1])
    return {'texts': texts, 'labels': labels}


def save_data(texts, labels, file, word_sep=' ', line_sep='\r\n', line_break='\r\n'):
    """
    :param texts:
    :param labels:
    :param file:
    :param word_sep: 字(词)与其标签之间的分隔符；
    :param line_sep: 在model为'ernie'时，对应文本和标签的分隔符；
    :param line_break: 不同文本间的分隔符；
    """
    assert len(texts) == len(labels)
    save_list = []
    for text, label in zip(texts, labels):
        for t, l in zip(text, label):
            save_list.append(word_sep.join([t, l]) + line_sep)
        save_list.append(line_break)
    with codecs.open(file, 'w', encoding='utf-8') as f:
        for l in save_list:
            f.write(l)


def load_all_texts(files):
    """
    将多个文件中的文本合并在一起
    """
    all_texts = []
    for file in files:
        data = load_data(file)
        all_texts.extend(data['texts'])
    return all_texts


def load_embedding(embedding_file):
    """
    加载词向量文件，并返回一个dict，其中key是词，val则是对应的向量
    """
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.strip().split(" ")) for o in tqdm(open(embedding_file, encoding='utf-8')))
    return embeddings_index


def build_matrix(embeddings_index, word_index):
    """
    构建词向量矩阵

    :param embeddings_index: dict, key是词，val是对应的向量
    :param word_index: dict，key是当前词表中的词，val是该词对应的id
    :return: numpy.ndarray, embedding矩阵
    """
    embedding_matrix = np.zeros((len(word_index), 300))
    for word, i in tqdm(word_index.items()):
        if i >= len(word_index):
            continue
        try:
            # word对应的vector
            embedding_vector = embeddings_index[word]
        except:
            # word不存在则使用unknown的vector
            embedding_vector = embeddings_index["未知"]
            # embedding_vector = embeddings_index["unknown"]
        if embedding_vector is not None:
            # 保证embedding_matrix行的向量与word_index中序号一致
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
