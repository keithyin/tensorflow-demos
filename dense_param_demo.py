
import tensorflow as tf

x = tf.random_uniform(shape=[2, 3])
x = tf.layers.dense(x, units=1, bias_regularizer=tf.contrib.layers.l2_regularizer(0.001))
if __name__ == '__main__':
    print(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    print(tf.get_collection(tf.GraphKeys.LOSSES))
    print(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

