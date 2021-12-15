import time

import tensorflow as tf
import numpy as np

SAMPLE_SIZE = 1000
SEQ_LEN = 100
EMB_TABLE_SIZE = 1000

cols = [
    tf.feature_column.categorical_column_with_identity(key="seq_{}".format(i),
                                                       num_buckets=EMB_TABLE_SIZE) for i in range(SEQ_LEN)]
shared_embs = tf.feature_column.shared_embedding_columns(cols, dimension=16)

np.random.seed(100)


def _parse(fea, label):
    features = {"seq_{}".format(i): fea[:, i:i + 1] for i in range(SEQ_LEN)}
    return features, label


def _parse_v2(fea, label):
    features = {"seq_{}".format(i): fea[:, i:i + 1] for i in range(SEQ_LEN)}
    concated_feas = []
    for i in range(SEQ_LEN):
        concated_feas.append(features["seq_{}".format(i)])
    concated_feas = tf.concat(concated_feas, axis=1)
    return concated_feas, label


def input_fn_builder(parse_fn):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices(
            (np.random.randint(0, EMB_TABLE_SIZE, size=SAMPLE_SIZE * SEQ_LEN).reshape([SAMPLE_SIZE, SEQ_LEN]),
             np.random.randint(0, 2, size=SAMPLE_SIZE).reshape([SAMPLE_SIZE, 1])))
        dataset = dataset.repeat(100).batch(16).map(lambda x, y: parse_fn(x, y)).prefetch(10)
        return dataset
    return input_fn


def model_fn(features, labels, mode):
    tf.logging.error("model_fn_1.build_graph_begin:{}".format(time.time()))

    x = tf.feature_column.input_layer(features, shared_embs)
    tf.logging.error("model_fn:{}".format(x))
    x = tf.layers.dense(x, units=256, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=128, activation=tf.nn.relu)
    logit = tf.layers.dense(x, units=1, activation=None)

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(labels, logit))
        train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(loss, global_step=tf.train.get_global_step())
        tf.logging.error("model_fn_1.build_graph_done:{}".format(time.time()))
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def model_fn_2(features, labels, mode):
    tf.logging.error("model_fn_2.build_graph_begin:{}".format(time.time()))
    emb_table = tf.get_variable(name="emb", shape=[EMB_TABLE_SIZE, 16], dtype=tf.float32)
    x = tf.nn.embedding_lookup(emb_table, features)
    x = tf.reshape(x, shape=[-1, np.prod(x.shape[1:])])
    tf.logging.error("model_fn_2:{}".format(x))
    x = tf.layers.dense(x, units=256, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=128, activation=tf.nn.relu)
    logit = tf.layers.dense(x, units=1, activation=None)

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(labels, logit))
        train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(loss, global_step=tf.train.get_global_step())
        tf.logging.error("model_fn_2.build_graph_done:{}".format(time.time()))
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    pass


def profile_1():
    # 1e5 samples. batch16. 129s
    estimator = tf.estimator.Estimator(model_fn=model_fn)
    begin_t = time.time()
    estimator.train(input_fn=input_fn_builder(_parse))
    tf.logging.error("profile1, end_time:{}".format(time.time()))
    tf.logging.error("profile1, delta:{}".format(time.time() - begin_t))


def profile_2():
    # 1e5 samples. batch16. 16s
    estimator = tf.estimator.Estimator(model_fn=model_fn_2)
    begin_t = time.time()
    estimator.train(input_fn=input_fn_builder(_parse_v2))
    tf.logging.error("profile2, end_time:{}".format(time.time()))
    tf.logging.error("profile2, delta:{}".format(time.time() - begin_t))


if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    profile_2()
    profile_1()

