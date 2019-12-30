import sys

from sklearn.linear_model import LogisticRegression, LinearRegression

from metalfi.src.controller import Controller
from metalfi.src.data.dataset import Dataset
from metalfi.src.data.memory import Memory
from metalfi.src.model.metamodel import MetaModel


class Main(object):

    @staticmethod
    def main():
        c = Controller()
        c.train_and_test()


if __name__ == '__main__':
    Main().main()
