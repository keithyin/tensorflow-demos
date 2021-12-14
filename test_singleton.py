from __future__ import print_function


class Singleton(object):
    def __new__(cls, name):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Singleton, cls).__new__(cls)
            return cls.instance
        else:
            raise ValueError("gg")

    def __init__(self, name):
        if not hasattr(self.instance, "__flag"):
            self.instance.__flag = True
            self.__class__.__inited = 10
            self.name = name

a = 10
def modify_a():
    global a
    if a == 10:
        a = 20


if __name__ == '__main__':
    s = Singleton("hello")
    print(s.name)
    modify_a()
    print(a)
