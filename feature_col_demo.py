import tensorflow as tf

visited_city = tf.feature_column.sequence_categorical_column_with_hash_bucket('visited_city', hash_bucket_size=1000)
visited_city2 = tf.feature_column.categorical_column_with_hash_bucket('visited_city2', hash_bucket_size=10)
visited_city3 = tf.feature_column.numeric_column("visited_val")
# visited_city4 = tf.feature_column.sequence_numeric_column('visited_city4')
feature_description = tf.feature_column.make_parse_example_spec([visited_city,
                                                                 visited_city2, visited_city3])

try:
    assert 1 == 2, "not good"
except Exception as e:
    raise ValueError(str(e) + "hhhhh")

