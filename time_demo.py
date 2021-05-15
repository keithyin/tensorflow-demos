import tensorflow as tf

label = tf.constant([1, 2, 3], dtype=tf.float32)
pred = tf.constant([2, 3, 4], dtype=tf.float32)
weight = tf.constant([0, 0, 0], dtype=tf.float32)

a = tf.losses.mean_squared_error(labels=label, predictions=pred, weights=weight, reduction=tf.losses.Reduction.MEAN)

with tf.Session() as sess:
    out = sess.run(a)
    print(out)

