import logging
import numpy as np
import shutil
import time
from utils import load_data
from enums import STRATEGY
from configs import ModelConfig, ActiveConfig
from model import Model
from networks import BiLstmCrf
from active_utils import DataPool, ActiveStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
text_lens = []


def start(train, dev, model_config, active_config, network):
    model = Model(model_config, network)
    selected_texts = None
    selected_labels = None

    logger.info("The active strategy is {}".format(active_config.select_strategy))
    if active_config.select_strategy == STRATEGY.RAND:
        dataPool = DataPool(train['texts'], train['labels'], 0)
        unselected_texts, unselected_labels = dataPool.get_unselected()
        tobe_selected_idx = ActiveStrategy.random_sampling(unselected_texts, active_config.total_num)
        dataPool.update(tobe_selected_idx)
        selected_texts, selected_labels = dataPool.get_selected()
        logger.info("The final size of selected data is {}".format(len(selected_texts)))
        if len(selected_texts) == 0:
            model.eval(dev['texts'], dev['texts'])
        else:
            model.train_and_eval(selected_texts, selected_labels, dev['texts'], dev['labels'])
    else:
        dataPool = DataPool(train['texts'], train['labels'], active_config.select_num)
        while (selected_texts is None) or len(selected_texts) < active_config.total_num - 5:
            selected_texts, selected_labels = dataPool.get_selected()
            unselected_texts, unselected_labels = dataPool.get_unselected()
            print("The size of selected data is {}".format(len(selected_texts)))
            print("The size of unselected data is {}".format(len(unselected_texts)))
            if len(selected_texts) != 0:
                model.train(selected_texts, selected_labels)
            if active_config.select_strategy == STRATEGY.LC:
                scores = model.predict_viterbi_score(unselected_texts)
                tobe_selected_idxs, tobe_selected_scores = ActiveStrategy.lc_sampling(scores, unselected_texts,
                                                                                      active_config.select_num)
            elif active_config.select_strategy == STRATEGY.MNLP:
                scores = model.predict_mnlp_score(unselected_texts)
                tobe_selected_idxs, tobe_selected_scores = ActiveStrategy.mnlp_sampling(scores, unselected_texts,
                                                                                        active_config.select_num)
            elif active_config.select_strategy == STRATEGY.TTE:
                probs = model.predict_probs(unselected_texts)
                tobe_selected_idxs, tobe_selected_scores = ActiveStrategy.tte_sampling(probs, unselected_texts,
                                                                                       active_config.select_num)
            elif active_config.select_strategy == STRATEGY.TE:
                probs = model.predict_probs(unselected_texts)
                tobe_selected_idxs, tobe_selected_scores = ActiveStrategy.te_sampling(probs, unselected_texts,
                                                                                      active_config.select_num)
            dataPool.update(tobe_selected_idxs)

        shutil.rmtree(model.config.model_dir)
        model_config.epochs = 20
        model = Model(model_config, network)
        print("The max size of selected data is {}".format(active_config.total_num))
        print("The size of selected data is {}".format(len(selected_texts)))
        lens = np.array([len(t) for t in selected_texts])
        text_lens.append(lens.mean())
        print("The size of unselected data is {}".format(len(unselected_texts)))
        model.train_and_eval(selected_texts, selected_labels, dev['texts'], dev['labels'])


if __name__ == '__main__':
    train_file = './data/train.txt'
    dev_file = './data/dev.txt'
    train = load_data(train_file)
    dev = load_data(dev_file)
    active_config = ActiveConfig(len(train['texts']))
    active_config.select_strategy = STRATEGY.RAND
    model_config = ModelConfig()
    model_config.epochs = 20
    for i in np.arange(1.0, 1.1, 0.2):
        i = round(i, 1)
        active_config.total_percent = i
        active_config.update()
        for j in range(3):
            model_config.model_dir = './results/model_CLUE/Random/model'+'_'+str(i)+'_'+str(j)
            start(train, dev, model_config, active_config, BiLstmCrf)
            time.sleep(5)
    print(text_lens)