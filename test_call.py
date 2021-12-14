class A(object):
    def __init__(self):
        self.a = 100

    def print(self):
        print(self.a)


if __name__ == '__main__':
    a = A()
    func = a.print
    func()