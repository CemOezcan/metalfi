from pathlib import Path

import numpy as np
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.datasets import load_wine, load_breast_cancer, load_iris, load_boston

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer


class Memory:

    def __init__(self):
        return

    @staticmethod
    def load(name):
        path = Memory.getPath()
        try:
            data = pd.read_csv(path / ("preprocessed/pp" + name)), True
        except FileNotFoundError:
            data = pd.read_csv(path / ("raw/" + name)), False

        return data

    @staticmethod
    def loadBoston():
        boston = load_boston()
        data_frame = DataFrame(data=boston.data, columns=boston['feature_names'])

        est = KBinsDiscretizer(n_bins=4, encode='ordinal')
        data_frame["target"] = est.fit_transform(list(map(lambda x: [x], boston.target)))

        return data_frame, "target"

    @staticmethod
    def loadTitanic():
        data_frame, preprocessed = Memory.load("titanic.csv")

        if not preprocessed:
            data_frame = data_frame.drop("Ticket", axis=1)

            data_frame["Sex"].replace({"male": 1, "female": 0}, inplace=True)

            data_frame["Name"] = [x.split(", ")[1] for x in data_frame["Name"]]
            data_frame["Name"] = [x.split(' ')[0] for x in data_frame["Name"]]
            data_frame["Name"] = data_frame["Name"].apply(
                lambda x: 0 if (x == "Mrs." or x == "Ms.") else (1 if x == "Mr." else 2))
            data_frame = data_frame.rename(columns={"Name": "Title"})

            data_frame["Cabin"] = [0 if (str(x) == "nan") else x[0] for x in data_frame["Cabin"]]
            data_frame["Cabin"].replace({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}, inplace=True)

            data_frame["Embarked"] = [0 if (str(x) == "nan") else x for x in data_frame["Embarked"]]
            data_frame["Embarked"].replace({'C': 1, 'Q': 2, 'S': 3}, inplace=True)

            avg_age = int(data_frame["Age"].mean())
            data_frame["Age"] = [avg_age if (str(x) == "nan") else x for x in data_frame["Age"]]

            data = {"Survived": data_frame["Survived"].values}
            data_frame_2 = DataFrame(data, columns=["Survived"])
            data_frame = data_frame.drop("Survived", axis=1)
            data_frame = data_frame.assign(Survived=data_frame_2["Survived"])
            data_frame = data_frame.drop("PassengerId", axis=1)

            data_frame.to_csv(Memory.getPath() / "preprocessed/pptitanic.csv", index=None, header=True)

        return data_frame, "Survived"

    @staticmethod
    def loadCancer():
        data_frame, preprocessed = Memory.load("cancer.csv")

        return data_frame.drop("Unnamed: 0", axis=1), "MEDV"

    @staticmethod
    def loadWine():
        wine = load_wine()
        data_frame = DataFrame(data=np.c_[wine['data'], wine['target']], columns=wine['feature_names'] + ['target'])

        return data_frame, "target"

    @staticmethod
    def loadIris():
        iris = load_iris()
        data_frame = DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

        return data_frame, "target"

    def storePreprocessed(self, data):
        return

    def storeInput(self, data):
        return

    def storeOutput(self, data):
        return

    def storeModel(self, data):
        return

    def storeVisual(self, data):
        return

    @staticmethod
    def getPath():
        path = Path(__file__).parents[2] / "data"
        return path
