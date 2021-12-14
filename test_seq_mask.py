from __future__ import print_function

import tensorflow as tf
import datetime

print(datetime.datetime.strptime("20211101", "%Y%m%d").weekday())


a = tf.constant([[2], [3]])
mask = tf.sequence_mask(a, maxlen=4)
b = tf.constant([-1])
b_res = tf.mod(b, 7)
if __name__ == '__main__':

    with tf.Session() as sess:
        print(sess.run(b_res))