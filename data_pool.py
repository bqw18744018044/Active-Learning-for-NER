import random
import numpy as np


class DataPool(object):
    def __init__(self, texts, labels, init_num):
        self.text_pool = np.array(texts)
        self.label_pool = np.array(labels)
        assert len(texts) == len(labels)
        self.pool_size = len(texts)
        # _l表示已标注数据集,_u表示未标注数据集
        self.train_texts = None
        self.train_labels = None
        self.select_texts = None
        self.select_labels = None
        self.train_idx = sorted(set(random.sample(list(range(self.pool_size)), init_num)))
        self.select_idx = sorted(set(range(self.pool_size)) - set(self.train_idx))
        self.update_pool()

    def update_pool(self):
        self.train_texts = self.text_pool[self.train_idx]
        self.train_labels = self.label_pool[self.train_idx]
        self.select_texts = self.text_pool[self.select_idx]
        self.select_labels = self.label_pool[self.select_idx]

    def update_idx(self, new_train_idx):
        new_train_idx = set(new_train_idx)
        self.train_idx = sorted(set(self.train_idx) | new_train_idx)
        self.select_idx = sorted(set(self.select_idx) - new_train_idx)

    def translate_select_idx(self, source_idx):
        target_idx = [self.select_idx[idx] for idx in source_idx]
        return target_idx

    def get_train(self):
        return self.train_texts, self.train_labels

    def get_select(self):
        return self.select_texts, self.select_labels
