import logging
import enum
from utils import load_data, save_data
from model import Model
from networks import BiLSTMCRF
from active_utils import DataPool, ActiveStrategy

# 枚举主动学习策略
STRATEGY = enum.Enum('STRATEGY', ('RAND', 'LC', 'MNLP', 'TTE', 'TE'))

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
               'dev': './data/dev.txt',
               'active': './data/active_train.txt'}  # 去掉
train_config = {'buffer_size': 15000,
                'epochs': 5,
                'batch_size': 32,
                'save_checkpoints_steps': 500,
                'model_dir': 'results/model'}
# 主动学习参数
active_config = {'initial_num': 3000,
                 'total_num': 10000,
                 'incremental_num': 1000,
                 'active_strategy': STRATEGY.TE}
config = {'model': model_config,
          'data': data_config,
          'train': train_config,
          'active': active_config}


train = load_data(data_config['train'])
dev = load_data(data_config['dev'])
logger.info("The size of train is {}".format(len(train['texts'])))
logger.info("The size of dev is {}".format(len(dev['texts'])))
dataPool = DataPool(train['texts'], train['labels'], active_config['initial_num'])
model = Model(config, BiLSTMCRF)
selected_texts = None
selected_labels = None

if active_config['active_strategy'] == STRATEGY.RAND:
    # random sampling
    unselected_texts, unselected_labels = dataPool.get_unselected()
    tobe_selected_idx = ActiveStrategy.random_sampling(unselected_texts,
                                                       active_config['total_num'] - active_config['initial_num'])
    dataPool.update(tobe_selected_idx)
    selected_texts, selected_labels = dataPool.get_selected()
    logger.info("The final size of selected data is {}".format(len(selected_texts)))
else:
    while (selected_texts is None) or len(selected_texts) < active_config['total_num']:
        selected_texts, selected_labels = dataPool.get_selected()
        unselected_texts, unselected_labels = dataPool.get_unselected()
        print("The size of selected data is {}".format(len(selected_texts)))
        print("The size of unselected data is {}".format(len(unselected_texts)))
        #logger.info("The size of selected data is {}".format(len(selected_texts)))
        #logger.info("The size of unselected data is {}".format(len(unselected_texts)))
        model.train(selected_texts, selected_labels)
        model.eval(dev['texts'], dev['labels'])
        if active_config['active_strategy'] == STRATEGY.LC:
            scores = model.predict_viterbi_score(unselected_texts)
            tobe_selected_idxs, tobe_selected_scores = ActiveStrategy.lc_sampling(scores, unselected_texts,
                                                                                  active_config['incremental_num'])
        elif active_config['active_strategy'] == STRATEGY.MNLP:
            scores = model.predict_viterbi_score(unselected_texts)
            tobe_selected_idxs, tobe_selected_scores = ActiveStrategy.mnlp_sampling(scores, unselected_texts,
                                                                                    active_config['incremental_num'])
        elif active_config['active_strategy'] == STRATEGY.TTE:
            probs = model.predict_probs(unselected_texts)
            tobe_selected_idxs, tobe_selected_scores = ActiveStrategy.tte_sampling(probs, unselected_texts,
                                                                                   active_config['incremental_num'])
        elif active_config['active_strategy'] == STRATEGY.TE:
            probs = model.predict_probs(unselected_texts)
            tobe_selected_idxs, tobe_selected_scores = ActiveStrategy.te_sampling(probs, unselected_texts,
                                                                                  active_config['incremental_num'])
        dataPool.update(tobe_selected_idxs)

# save_data(selected_texts, selected_labels, './data/selected_train.txt')


