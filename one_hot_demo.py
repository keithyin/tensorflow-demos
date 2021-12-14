import tensorflow as tf
if __name__ == '__main__':
    a = tf.constant([[1], [2], [3]], dtype=tf.int64)
    one_hotted = tf.one_hot(a, depth=4)
    print(one_hotted.shape)
    with tf.Session() as sess:
        print(sess.run(one_hotted))