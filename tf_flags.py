
import tensorflow as tf

tf.flags.DEFINE_string("arg1", default="???", help="")
FLAGS = tf.flags.FLAGS

def main(_):
    print("arg1: [{}].".format(FLAGS.arg1))
    pass


if __name__ == '__main__':
    tf.app.run()