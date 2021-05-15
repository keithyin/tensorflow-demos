import tensorflow as tf


sess = tf.Session()

# 特征数据
features = {
    'birthplace': [[1, 2], [1, 4], [3, 3], [4, 5]],
    'age': [[1], [2], [3], [4]],
    'profile_list': [[1, 2], [1, 4], [3, 3], [4, 5]]
}

# 特征列
birthplace = tf.feature_column.categorical_column_with_identity("birthplace", num_buckets=3, default_value=0)
age = tf.feature_column.categorical_column_with_identity("age", num_buckets=3, default_value=0)
birthplace_onehot = tf.feature_column.indicator_column(birthplace)
birthplace_emb = tf.feature_column.embedding_column(birthplace, dimension=3)
shared_emb = tf.feature_column.shared_embedding_columns([birthplace, age], 3)
profile_list = tf.feature_column.sequence_numeric_column('profile_list', shape=2)
profile_list = tf.feature_column.bucketized_column(profile_list, boundaries=[1, 2, 3])
# 组合特征列
columns = [
    birthplace_onehot, birthplace_emb, shared_emb[0], shared_emb[1], profile_list
]

print(tf.feature_column.make_parse_example_spec(columns))

# 输入层（数据，特征列）
inputs = tf.feature_column.input_layer(features, columns)

# 初始化并运行
init = tf.global_variables_initializer()
sess.run(tf.tables_initializer())
sess.run(init)
v = sess.run(inputs)
print(v)
with tf.Session() as sess:
    sess.run()
    tf.parse_single_example()