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
        self.embed = './data/sgn_renmin.npy'
        self.vocab = './data/vocab.txt'
        self.tags = ['B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-BOOK', 'I-BOOK', 'B-COMP', 'I-COMP',
                      'B-GAME', 'I-GAME', 'B-GOVERN', 'I-GOVERN', 'B-MOVIE', 'I-MOVIE', 'B-POS', 'I-POS', 'B-SCENE',
                      'I-SCENE', 'O', 'X', '[CLS]', '[SEP]']
        self.buffer_size = 15000
        self.epochs = 20
        self.batch_size = 32
        self.save_checkpoints_steps = 500
        self.model_dir = './results/model_CLUE'


class ActiveConfig(object):
    def __init__(self, pool_size, total_percent=0.5, select_percent=0.25):
        self.pool_size = pool_size
        self.total_percent = total_percent
        self.select_percent = select_percent
        self.total_num = None
        self.select_num = None
        self.select_strategy = STRATEGY.MNLP
        self.update()

    def update(self):
        self.total_num = int(self.pool_size*self.total_percent)
        self.select_num = int(self.total_num*self.select_percent)
