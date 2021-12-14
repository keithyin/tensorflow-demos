# coding=utf-8
import tensorflow as tf
from tensorflow import contrib

if __name__ == '__main__':

    # 特征数据
    features = {
        'birthplace': tf.constant([[1], [4], [3], [5]]),
        'age': tf.constant([[1], [2], [3], [4]]),
        'profile_list': tf.constant([[1, 2], [1, 4], [3, 3], [4, 5]])
    }

    # 特征列
    birthplace = tf.feature_column.sequence_categorical_column_with_identity("birthplace", num_buckets=3, default_value=0)
    # age = tf.feature_column.categorical_column_with_identity("age", num_buckets=3, default_value=0)
    birthplace_onehot = tf.feature_column.indicator_column(birthplace)
    birthplace_emb = tf.feature_column.embedding_column(birthplace, dimension=3)
    # shared_emb = tf.feature_column.shared_embedding_columns([birthplace, age], 3)
    # profile_list = tf.feature_column.numeric_column('profile_list', shape=[2])
    # profile_list = tf.feature_column.bucketized_column(profile_list, boundaries=[1, 2, 3])
    # 组合特征列
    columns = [
        birthplace_onehot, birthplace_emb, # shared_emb[0], shared_emb[1], profile_list
    ]

    print(tf.feature_column.make_parse_example_spec(columns))

    # 输入层（数据，特征列）
    inputs = contrib.feature_column.sequence_input_layer(features, columns)

    # 初始化并运行
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(tf.tables_initializer())
        sess.run(init)
        v = sess.run(inputs)
        print(v[0].shape)
