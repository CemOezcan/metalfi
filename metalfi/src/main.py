from metalfi.src.controller import Controller


class Main(object):

    @staticmethod
    def main():
        c = Controller()
        c.trainMetaModel()


if __name__ == '__main__':
    Main().main()
