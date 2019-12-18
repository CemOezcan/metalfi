import pickle

from pathlib import Path
from pandas import DataFrame


class MetaModel:

    def __init__(self, model, train, test, name):
        self.__model = model
        self.__test_data = test
        self.__train_data = train
        self.__meta_data = DataFrame()
        self.file_name = name

    def save(self):
        # TODO: Implement in Memory class
        pickle.dump(self.__model, open(Path(__file__).parents[2] / ("data/model/" + self.file_name), 'wb'))

    def run(self):
        return
