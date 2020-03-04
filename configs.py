from enums import STRATEGY


class ModelConfig(object):
    def __init__(self):
        self.embed_dim = 300
        self.dropout = 0.5
        self.lstm_size = 256
        self.lstm_layer = 1
        self.vocab_size = 30000
        self.use_pretrained = True
        self.num_oov_buckets = 1
        self.embed = './data/chinese/sgn_renmin.npy'
        self.vocab = './data/chinese/vocab.txt'
        self.positive_tags = ['B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-BOOK', 'I-BOOK', 'B-COMP',
                              'I-COMP', 'B-GAME', 'I-GAME', 'B-GOVERN', 'I-GOVERN', 'B-MOVIE', 'I-MOVIE', 'B-POS',
                              'I-POS', 'B-SCENE', 'I-SCENE']
        self.positive_ids = None
        self.tags = None
        self.buffer_size = 15000
        self.epochs = 20
        self.batch_size = 32
        self.save_checkpoints_steps = 500
        self.model_dir = './results/model_chinese'
        self.update()

    def update(self):
        """
        When you reset any parameters, call this method to update the relevant parameters.
        """
        self.positive_ids = list(range(len(self.positive_tags)))
        self.tags = self.positive_tags + ['O', 'X', '[CLS]', '[SEP]']


class ActiveConfig(object):
    def __init__(self, pool_size, total_percent=0.5, select_percent=0.25, select_epochs=10, total_epochs=20):
        self.pool_size = pool_size
        self.total_percent = total_percent
        self.select_percent = select_percent
        self.total_num = None
        self.select_num = None
        self.select_strategy = STRATEGY.MNLP
        self.select_epochs = select_epochs  # the train epochs each time new samples are added
        self.total_epochs = total_epochs  # the train epochs from scratch when finish sampling
        self.update()

    def update(self):
        """
        When you reset any parameters, call this method to update the relevant parameters.
        """
        self.total_num = int(self.pool_size*self.total_percent)
        self.select_num = int(self.total_num*self.select_percent)
