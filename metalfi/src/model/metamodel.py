import pickle

from pathlib import Path
from pandas import DataFrame
from sklearn.multioutput import MultiOutputRegressor

from metalfi.src.data.metadataset import MetaDataset


class MetaModel:

    def __init__(self, model, train, test, name):
        # TODO: Parameter optimization?
        self.__model = model
        self.__test_data = MetaDataset(test)
        self.__train_data = MetaDataset(train, True)
        self.file_name = name

    def save(self):
        # TODO: Implement in Memory class
        pickle.dump(self.__model, open(Path(__file__).parents[2] / ("data/model/" + self.file_name), 'wb'))

    def run(self):
        targets = self.__train_data.getTargetNames()
        y = self.__train_data.getMetaData()[targets]
        X = self.__train_data.getMetaData().drop(targets, axis=1)

        self.__model.fit(X, y)

        print(y)
        print(self.__train_data.getMetaData().index)
        print(self.__model.predict(X))
