from metalfi.src.data.memory import Memory


class Main(object):

    @staticmethod
    def main():
        Memory.loadTitanic()


if __name__ == '__main__':
    Main().main()
