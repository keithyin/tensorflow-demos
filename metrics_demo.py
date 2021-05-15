import sklearn.metrics as sk_metrics
import numpy as np
import tensorflow as tf


a1 = tf.random.uniform(shape=[100], minval=0, maxval=1)
a2 = tf.random.uniform(shape=[100], minval=0, maxval=1)

b1 = tf.constant(np.random.random_integers(0, 1, size=[100]), dtype=tf.float32)
b2 = tf.constant(np.random.random_integers(0, 1, size=[100]), dtype=tf.float32)

tf_auc = tf.metrics.AUC(num_thresholds=1024)
tf_auc.update_state(b1, a1)
tf_auc.update_state(b2, a2)
print(tf_auc.result())
print(sk_metrics.roc_auc_score(np.concatenate([b1.numpy(), b2.numpy()]), np.concatenate([a1.numpy(), a2.numpy()])))
b1a1 = sk_metrics.roc_auc_score(b1.numpy(), a1.numpy())
b2a2 = sk_metrics.roc_auc_score(b2.numpy(), a2.numpy())
print(b1a1)
print(b2a2)
print((b1a1+b2a2) / 2)


