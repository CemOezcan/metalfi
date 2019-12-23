import sys

from sklearn.linear_model import LogisticRegression, LinearRegression

from metalfi.src.data.dataset import Dataset
from metalfi.src.data.memory import Memory
from metalfi.src.model.metamodel import MetaModel


class Main(object):

    @staticmethod
    def main():
        data_frame, target, name = Memory.loadTitanic()
        data = Dataset(data_frame, target, name)

        data_frame_2, target_2, name = Memory.loadCancer()
        data_2 = Dataset(data_frame_2, target_2, name)

        data_frame_3, target_3, name = Memory.loadIris()
        data_3 = Dataset(data_frame_3, target_3, name)

        data_frame_4, target_4, name = Memory.loadWine()
        data_4 = Dataset(data_frame_4, target_4, name)

        model = MetaModel([data_2, data_3, data_4, data], [data_3], "name")
        model.run()


if __name__ == '__main__':
    Main().main()
