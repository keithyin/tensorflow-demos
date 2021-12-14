import numpy as np
import tensorflow as tf
a = tf.constant([["1", "1"], ["2", "2"]], dtype=tf.string)
b = tf.constant([["2", "2"], ["3", "3"]], dtype=tf.string)

a_b = tf.string_join([a, b], separator="_")
a_b = tf.string_to_hash_bucket_strong(a_b, num_buckets=2**63-1, key=[0, 0])

c = tf.constant([1, 2], dtype=tf.int64)
c_ = tf.strings.as_string(c, precision=3)

if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run([a_b, c_]))
