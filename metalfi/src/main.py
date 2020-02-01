from metalfi.src.controller import Controller


class Main(object):

    @staticmethod
    def main():
        c = Controller()
        c.storeMetaData()


if __name__ == '__main__':
    Main().main()
