import tensorflow as tf
import numpy as np
import functools
from pathlib import Path
from tf_metrics import precision, recall, f1


class Model(object):
    def __init__(self, config, network):
        self.config = config
        self.network = network
        cfg = tf.estimator.RunConfig(save_checkpoints_steps=self.config['train']['save_checkpoints_steps'])  # 用于指定estimator运行的参数
        self.estimator = tf.estimator.Estimator(self.model_fn, self.config['train']['model_dir'], cfg, self.config['model'])

    def input_fn(self, texts, labels, params=None, shuffle_and_repeat=False):
        def generator_fn():
            for text, label in zip(texts, labels):
                assert len(text) == len(label)
                # 按estimator的约定，返回包含两个元素的元组， 第一个作为features，第二个作为labels
                yield (text, len(text)), label
        params = params if params is not None else {}
        # None表示不确定,()表示只有1个数
        # shape分别对应generator_fn返回数据的形状
        shapes = (([None], ()), [None])
        types = ((tf.string, tf.int32), tf.string)
        defaults = (('<pad>', 0), 'O')

        dataset = tf.data.Dataset.from_generator(functools.partial(generator_fn),
                                                 output_shapes=shapes,
                                                 output_types=types)
        if shuffle_and_repeat:
            dataset = dataset.shuffle(params['buffer_size']).repeat(params['epochs'])

        # 进行padding，其中padding的长度由shape来指定(为None是按最长文本进行padding)，padding的值由defaults指定
        # padded_batch是对batch进行padding，因此每个batch最终的长度可能不一致
        dataset = (dataset.padded_batch(params.get('batch_size', 32), shapes, defaults).prefetch(1))
        return dataset

    def model_fn(self, features, labels, mode, params):
        # 在estimator中，由dataset产生的数据是一个包含两个元素的元组，
        # 其中第一个元素指定为features，第二个元素指定为labels
        # training = (mode == tf.estimator.ModeKeys.TRAIN)
        net = self.network(features, labels, mode, params)
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {'probs': net.probs,
                           'pred_ids': net.pred_ids,
                           'tags': net.pred_strings,
                           'score': net.score}
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        else:
            metrics = {
                'acc': tf.metrics.accuracy(net.tags, net.pred_ids, net.weights),
                # [0,1,2,3,4,5]对应'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG'
                'precision': precision(net.tags, net.pred_ids, net.num_tags, [0, 1, 2, 3, 4, 5], net.weights),
                'recall': recall(net.tags, net.pred_ids, net.num_tags, [0, 1, 2, 3, 4, 5], net.weights),
                'f1': f1(net.tags, net.pred_ids, net.num_tags, [0, 1, 2, 3, 4, 5], net.weights),
            }
            for metric_name, op in metrics.items():
                tf.summary.scalar(metric_name, op[1])

            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode, loss=net.loss, eval_metric_ops=metrics)

            elif mode == tf.estimator.ModeKeys.TRAIN:
                train_op = tf.train.AdamOptimizer().minimize(
                    net.loss, global_step=tf.train.get_or_create_global_step())
                return tf.estimator.EstimatorSpec(mode, loss=net.loss, train_op=train_op)

    def train(self, texts, labels):
        inpf = functools.partial(self.input_fn, texts, labels, self.config['train'], shuffle_and_repeat=True)
        Path(self.estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
        self.estimator.train(input_fn=inpf)

    def eval(self, texts, labels):
        inpf = functools.partial(self.input_fn, texts, labels)
        self.estimator.evaluate(input_fn=inpf)

    def train_and_eval(self, tr_texts, tr_labels, dev_texts, dev_labels):
        train_inpf = functools.partial(self.input_fn, tr_texts, tr_labels, self.config['train'], shuffle_and_repeat=True)
        eval_inpf = functools.partial(self.input_fn, dev_texts, dev_labels)
        train_spec = tf.estimator.TrainSpec(input_fn=train_inpf)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=60)
        tf.estimator.train_and_evaluate(self.estimator, train_spec, eval_spec)

    def _predict(self, texts, labels=None):
        # 尽管预测时不需要标签，但是由于input_fn需要labels，因此如果labels为None，那么就生成一个labels
        if not labels:
            labels = []
            for text in texts:
                labels.append(['O'] * len(text))
        inpf = functools.partial(self.input_fn, texts, labels)
        preds = self.estimator.predict(inpf)
        preds = [pred for pred in preds]
        return preds

    def predict_tags(self, texts, labels=None):
        preds = self._predict(texts, labels)
        tags = [pred['tags'] for pred in preds]
        return tags

    def predict_viterbi_score(self, texts, labels=None):
        preds = self._predict(texts, labels)
        scores = [pred['score'] for pred in preds]
        return scores

    def predict_probs(self, texts, labels=None):
        preds = self._predict(texts, labels)
        probs = [pred['probs'] for pred in preds]
        return probs





