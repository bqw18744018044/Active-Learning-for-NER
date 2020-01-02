
class DataPool(object):
    def __init__(self, texts, labels, init_num):
        self.textpool = texts
        self.labelpool = labels
        self.init_num = init_num
        assert len(texts) == len(labels)
        self.pool_size = len(texts)
        # _l表示已标注数据集,_u表示未标注数据集
        self.texts_l = None
        self.labels_l = None
        self.texts_u = None
        self.labels_u = None