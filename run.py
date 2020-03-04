import logging
from utils import load_data
from enums import STRATEGY
from configs import ModelConfig, ActiveConfig
from networks import BiLstmCrf
from active_learning import start_active_learning

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_chinese():
    train_file = './data/chinese/train.txt'
    dev_file = './data/chinese/dev.txt'
    train = load_data(train_file)
    dev = load_data(dev_file)
    active_config = ActiveConfig(len(train['texts']))
    active_config.select_strategy = STRATEGY.TTE
    active_config.total_percent = 0.5
    active_config.update()

    model_config = ModelConfig()
    model_config.embed = './data/chinese/sgn_renmin.npy'
    model_config.vocab = './data/chinese/vocab.txt'
    model_config.positive_tags = ['B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-BOOK', 'I-BOOK', 'B-COMP',
                                  'I-COMP', 'B-GAME', 'I-GAME', 'B-GOVERN', 'I-GOVERN', 'B-MOVIE', 'I-MOVIE', 'B-POS',
                                  'I-POS', 'B-SCENE', 'I-SCENE']
    model_config.model_dir = './results/model_chinese'
    model_config.update()
    start_active_learning(train, dev, model_config, active_config, BiLstmCrf)


def run_english():
    train_file = './data/english/train.txt'
    dev_file = './data/english/test.txt'
    train = load_data(train_file)
    dev = load_data(dev_file)
    active_config = ActiveConfig(len(train['texts']))
    active_config.select_strategy = STRATEGY.MNLP
    active_config.total_percent = 0.5
    active_config.update()

    model_config = ModelConfig()
    model_config.embed = './data/english/glove_conll2003.npy'
    model_config.vocab = './data/english/vocab.txt'
    model_config.positive_tags = ['B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC']
    model_config.model_dir = './results/model_english'
    model_config.update()
    start_active_learning(train, dev, model_config, active_config, BiLstmCrf)


if __name__ == '__main__':
    #run_chinese()
    run_english()