from __future__ import print_function

from collections import namedtuple

A = namedtuple("A", ["a"])
a = A([1])
if __name__ == '__main__':
    print(a)
    a.a.append(2)
    print(a)
