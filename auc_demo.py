import tensorflow as tf
from sklearn.metrics import roc_auc_score
import numpy as np

predictions_ph = tf.placeholder(dtype=tf.float32, shape=[None])
labels_ph = tf.placeholder(dtype=tf.float32, shape=[None])

global_step = tf.train.get_or_create_global_step()
global_step = tf.assign_add(global_step, 1)

auc_res = tf.metrics.auc(labels=labels_ph, predictions=predictions_ph, num_thresholds=2048)

is_reset_global_step = tf.equal(global_step % 4, 0)

reset_auc = tf.cond(is_reset_global_step,
                    lambda: tf.group([tf.assign(ref, tf.zeros_like(ref))
                                      for ref in tf.local_variables() if 'auc' in ref.op.name]),
                    lambda: tf.no_op())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    tot_pred = []
    tot_label = []
    for i in range(16):
        i += 1
        pred = np.random.uniform(0, 1, size=[100])
        label = np.array([1] * i + [0] * (100 - i))
        tf_auc_res = sess.run([auc_res, reset_auc, global_step, is_reset_global_step],
                              feed_dict={predictions_ph: pred, labels_ph: label})
        print("tf_auc:", tf_auc_res)
        if tf_auc_res[1]:
            tot_pred = []
            tot_label = []
        tot_pred.append(pred)
        tot_label.append(label)
        print("batch:{:.4f}, tot:{:.4f}".format(roc_auc_score(label, pred),
                                                roc_auc_score(np.concatenate(tot_label, axis=0),
                                                              np.concatenate(tot_pred, axis=0))))

    # tot_label = np.concatenate(tot_label, axis=0)
    # tot_pred = np.concatenate(tot_pred, axis=0)
    # print("----")
    # print("sklean_roc_auc: ", roc_auc_score(tot_label, tot_pred))
