import tensorflow as tf
import logging
from utils import load_data
from model import Model
from networks import BiLSTMCRF

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 参数
model_config = {'dim': 300,  # 词向量的维度
                'dropout': 0.5,
                'lstm_size': 256,  # lstm输出的维度
                'lstm_layers': 1,  # 多层lstm的层数
                'vocab_size': 30000,
                'use_pretrained': True,
                'num_oov_buckets': 1,
                'embed': './data/sgn_renmin.npy',
                'vocab': './data/vocab.txt',
                'tags': ['B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'O']}
data_config = {'train': './data/train.txt',
               'dev': './data/dev.txt'}
train_config = {'buffer_size': 15000,
                'epochs': 1,
                'batch_size': 32,
                'save_checkpoints_steps': 100,
                'model_dir': 'results/model'}
config = {'model': model_config,
          'data': data_config,
          'train': train_config}

train = load_data(data_config['train'])
dev = load_data(data_config['dev'])
model = Model(config, BiLSTMCRF)
scores = model.predict_viterbi_score(dev['texts'], dev['labels'])
#preds = model.predict(dev_data['texts'], dev_data['labels'])
#model.train_and_eval(train_data['texts'], train_data['labels'], dev_data['texts'], dev_data['labels'])
#model.train(train_data['texts'], train_data['labels'])
#model.eval(dev_data['texts'], dev_data['labels'])
