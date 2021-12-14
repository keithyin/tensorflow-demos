from __future__ import print_function
import tensorflow as tf
from tensorflow.python.training import monitored_session


def err_func():
    raise ValueError("gg")


if __name__ == '__main__':
    try:
        raise err_func()
    finally:
        print("hello")
