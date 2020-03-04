import logging
import numpy as np
import shutil
from enums import STRATEGY
from model import Model
from active_utils import DataPool, ActiveStrategy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def start_active_learning(train, dev, model_config, active_config, network):
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
        model.config.epochs = active_config.total_epochs
        model.config.update()
        if len(selected_texts) == 0:
            model.eval(dev['texts'], dev['texts'])
        else:
            model.train_and_eval(selected_texts, selected_labels, dev['texts'], dev['labels'])
    else:
        dataPool = DataPool(train['texts'], train['labels'], active_config.select_num)
        model.config.epochs = active_config.select_epochs
        model.config.update()
        while (selected_texts is None) or len(selected_texts) < active_config.total_num - 5:
            selected_texts, selected_labels = dataPool.get_selected()
            unselected_texts, unselected_labels = dataPool.get_unselected()
            logger.info("The size of selected data is {}".format(len(selected_texts)))
            logger.info("The size of unselected data is {}".format(len(unselected_texts)))
            logger.info("Query strategy is {}".format(active_config.select_strategy))
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
        model_config.epochs = active_config.total_epochs
        model = Model(model_config, network)
        logger.info("The max size of selected data is {}".format(active_config.total_num))
        logger.info("The size of selected data is {}".format(len(selected_texts)))
        logger.info("The size of unselected data is {}".format(len(unselected_texts)))
        model.train_and_eval(selected_texts, selected_labels, dev['texts'], dev['labels'])