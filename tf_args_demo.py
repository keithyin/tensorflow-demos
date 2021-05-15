from __future__ import print_function
import tensorflow as tf
FLAG = tf.flags.FLAGS

tf.flags.DEFINE_integer(name="a", default=10, help="pass it")
print("before passed:", FLAG.a)


def main(_):
    print("parsed:", FLAG.a)


if __name__ == '__main__':
    tf.app.run()
