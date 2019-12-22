from sklearn.linear_model import LogisticRegression, LinearRegression

from metalfi.src.data.dataset import Dataset
from metalfi.src.data.memory import Memory
from metalfi.src.model.metamodel import MetaModel


class Main(object):

    @staticmethod
    def main():
        data_frame, target = Memory.loadTitanic()
        data = Dataset(data_frame, target)

        data_frame_2, target_2 = Memory.loadCancer()
        data_2 = Dataset(data_frame_2, target_2)

        model = LinearRegression()

        model = MetaModel(model, [data_2], [data], "name")
        model.run()


if __name__ == '__main__':
    Main().main()
