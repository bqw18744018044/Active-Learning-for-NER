from keras_preprocessing.text import Tokenizer
import numpy as np
import logging
import os
from utils import build_matrix, load_embedding, load_all_texts

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def clip_embedding_matrix(embedding_file, input_files, output_dir, embedding_name):
    vocab_file = os.path.join(output_dir, 'vocab.txt')
    clipped_file = os.path.join(output_dir, embedding_name)

    # load all files and build the vocabulary
    all_texts = load_all_texts(input_files)
    tokenizer = Tokenizer(num_words=None, lower=False)
    tokenizer.fit_on_texts(all_texts)
    logger.info("the size of vocabulary is {}".format(len(tokenizer.word_counts)))

    # load word vector and build embedding matrix
    embeddings_index = load_embedding(embedding_file)
    embedding_matrix = build_matrix(embeddings_index, tokenizer.word_index)
    logger.info("the shape of embedding matrix is {}".format(embedding_matrix.shape))

    # save embedding matrix and vocabulary
    np.save(clipped_file, embedding_matrix)  # save embedding matrix
    # save vocabulary
    words = [word + '\n' for word in list(tokenizer.word_index.keys())]
    with open(vocab_file, 'w', encoding='utf-8') as f:
        f.writelines(words)


def chinese_clip():
    EMBEDDING_FILE = './data/chinese/sgns.renmin.bigram-char'
    train_file = './data/chinese/train.txt'
    dev_file = './data/chinese/dev.txt'
    output_dir = './data/chinese'
    embedding_name = 'sgn_renmin'
    clip_embedding_matrix(EMBEDDING_FILE, [train_file, dev_file], output_dir, embedding_name)


def english_clip():
    EMBEDDING_FILE = './data/english/glove.840B.300d.txt'
    train_file = './data/english/train.txt'
    test_file = './data/english/test.txt'
    valid_file = './data/english/valid.txt'
    output_dir = './data/english'
    embedding_name = 'glove_conll2003'
    clip_embedding_matrix(EMBEDDING_FILE, [train_file, test_file, valid_file], output_dir, embedding_name)


if __name__ == '__main__':
    english_clip()
    #chinese_clip()

