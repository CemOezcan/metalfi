from metalfi.src.controller import Controller


class Main(object):

    @staticmethod
    def main():
        c = Controller()
        c.train_and_test()


if __name__ == '__main__':
    Main().main()
