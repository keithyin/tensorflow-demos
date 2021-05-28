import tensorflow as tf
a = tf.constant([[1], [1], [2]], dtype=tf.int64)
b = tf.constant([[1], [1], [2]], dtype=tf.int64)

res = tf.sparse_tensor_to_dense(tf.sparse.cross_hashed([a]))
if __name__ == '__main__':
    with tf.Session() as sess:
        print(sess.run(res))