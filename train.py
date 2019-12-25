import tensorflow as tf
import logging
import functools
from utils import load_data
from pathlib import Path
from model import BiLSTMCRF
from tf_metrics import precision, recall, f1

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
                'epochs': 30,
                'batch_size': 32,
                'save_checkpoints_steps': 1000,
                'model_dir': 'results/model'}
config = {'model': model_config,
          'data': data_config,
          'train': train_config}


# 定义input_fn
def generator_fn(data):
    texts = data['texts']
    labels = data['labels']
    for text, label in zip(texts, labels):
        assert len(text) == len(label)
        # 按estimator的约定，返回包含两个元素的元组， 第一个作为features，第二个作为labels
        yield (text, len(text)), label


def input_fn(data, params=None, shuffle_and_repeat=False):
    params = params if params is not None else {}
    # None表示不确定,()表示只有1个数
    # shape分别对应generator_fn返回数据的形状
    shapes = (([None], ()), [None])
    types = ((tf.string, tf.int32), tf.string)
    defaults = (('<pad>', 0), 'O')

    dataset = tf.data.Dataset.from_generator(functools.partial(generator_fn, data),
                                             output_shapes=shapes,
                                             output_types=types)
    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer_size']).repeat(params['epochs'])

    # 进行padding，其中padding的长度由shape来指定(为None是按最长文本进行padding)，padding的值由defaults指定
    # padded_batch是对batch进行padding，因此每个batch最终的长度可能不一致
    dataset = (dataset.padded_batch(params.get('batch_size', 32), shapes, defaults).prefetch(1))
    return dataset


def model_fn(features, labels, mode, params):
    # 在estimator中，由dataset产生的数据是一个包含两个元素的元组，
    # 其中第一个元素指定为features，第二个元素指定为labels
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    model = BiLSTMCRF(features, labels, training, params)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'pred_ids': model.pred_ids,
                       'tags': model.pred_strings}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        metrics = {
            'acc': tf.metrics.accuracy(model.tags, model.pred_ids, model.weights),
            # [0,1,2,3,4,5]对应'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG'
            'precision': precision(model.tags, model.pred_ids, model.num_tags, [0, 1, 2, 3, 4, 5], model.weights),
            'recall': recall(model.tags, model.pred_ids, model.num_tags, [0, 1, 2, 3, 4, 5], model.weights),
            'f1': f1(model.tags, model.pred_ids, model.num_tags, [0, 1, 2, 3, 4, 5], model.weights),
        }
        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=model.loss, eval_metric_ops=metrics)

        elif mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer().minimize(
                model.loss, global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(mode, loss=model.loss, train_op=train_op)


def train():
    # 加载数据
    train_data = load_data(data_config['train'])
    dev_data = load_data(data_config['dev'])
    train_inpf = functools.partial(input_fn, train_data, train_config, shuffle_and_repeat=True)
    eval_inpf = functools.partial(input_fn, dev_data)
    """
    # 测试input_fn
    dataset = train_inpf()
    iterator = dataset.make_one_shot_iterator()  # 创建迭代器
    next_element = iterator.get_next()
    with tf.Session() as sess:
        batch_data = sess.run(next_element)
        (text, text_len), label = batch_data
        logger.info("text:\n {}".format(text))
        logger.info("text_len:\n {}".format(text_len))
        logger.info("label:\n {}".format(label))
    """
    cfg = tf.estimator.RunConfig(save_checkpoints_steps=train_config['save_checkpoints_steps'])  # 用于指定estimator运行的参数
    estimator = tf.estimator.Estimator(model_fn, train_config['model_dir'], cfg, model_config)
    Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
    #hook = tf.contrib.estimator.stop_if_no_increase_hook(estimator, 'f1', 500, min_steps=8000, run_every_secs=120)
    #train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=60)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    train()
