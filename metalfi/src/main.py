from metalfi.src.data.dataset import Dataset
from metalfi.src.data.memory import Memory


class Main(object):

    @staticmethod
    def main():
        data_frame, target = Memory.loadTitanic()
        Dataset(data_frame, target)


if __name__ == '__main__':
    Main().main()
