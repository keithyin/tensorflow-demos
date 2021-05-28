# coding=utf-8
import tensorflow as tf


def _bytes_feature(*value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[*value]))


def _float_feature(*value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[*value]))


def _int64_feature(*value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[*value]))


def write_tf_record():
    with tf.python_io.TFRecordWriter("tfrecord.pb") as writer:
        for i in range(3):
            features = {
                "visited_city": _int64_feature(*([i] * (i + 1))),
                "sequence_feas": _int64_feature(*[1, 2, 3, 4])
            }
            example_proto = tf.train.Example(features=tf.train.Features(feature=features))
            # tf.train.SequenceExample
            writer.write(example_proto.SerializeToString())


def parse_from_record_graph():
    record_filename = tf.placeholder(dtype=tf.string, shape=[1])
    raw_image_dataset = tf.data.TFRecordDataset(record_filename)

    # Create a dictionary describing the features.
    feature_description = {
        'visited_city': tf.io.VarLenFeature(dtype=tf.int64),
        'sequence_feas': tf.io.FixedLenSequenceFeature(shape=[4], dtype=tf.int64, allow_missing=True)
    }

    def _parse_example_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        example = tf.parse_single_example(example_proto, feature_description)
        # visited_city = tf.io.decode_raw(example['visited_city'], out_type=tf.int64)
        visited_city = example["visited_city"]  # 因为使用 VarLenFeature解析，所以返回的是 tf.sparse.SparseTensor类型
        visited_city = tf.sparse.to_dense(visited_city)  # 转成 dense 类型。
        return visited_city, example['sequence_feas']

    parsed_image_dataset = raw_image_dataset.map(_parse_example_function)
    parsed_image_dataset = parsed_image_dataset.batch(1)
    iterator = parsed_image_dataset.make_initializable_iterator()

    next_val = iterator.get_next()
    #
    with tf.Session() as sess:
        sess.run(iterator.initializer, feed_dict={record_filename: ["tfrecord.pb"]})
        while True:
            print(sess.run(next_val))  # 这边出来的是 SparsedTensor。如果不想使用 SparseTensor，可以使用第一种方式。


if __name__ == '__main__':
    write_tf_record()
    parse_from_record_graph()