from pathlib import Path

from sklearn import preprocessing
from sklearn.datasets import load_wine, load_breast_cancer

import pandas as pd


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
    def loadTitanic():
        data_frame, preprocessed = Memory.load("titanic.csv")

        if not preprocessed:
            data_frame["Sex"].replace({"male": 1, "female": 0}, inplace=True)

            data_frame["Name"] = [x.split(", ")[1] for x in data_frame["Name"]]
            data_frame["Name"] = [x.split(' ')[0] for x in data_frame["Name"]]
            data_frame["Name"] = data_frame["Name"].apply(lambda x: 0 if (x == "Mrs." or x == "Ms.") else (1 if x == "Mr." else 2))
            data_frame = data_frame.rename(columns={"Name": "Title"})

            data_frame["Cabin"] = [0 if (str(x) == "nan") else x[0] for x in data_frame["Cabin"]]
            data_frame["Cabin"].replace({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8}, inplace=True)

            data_frame["Embarked"] = [0 if (str(x) == "nan") else x for x in data_frame["Embarked"]]
            data_frame["Embarked"].replace({'C': 1, 'Q': 2, 'S': 3}, inplace=True)

            avg_age = int(data_frame['Age'].mean())
            data_frame["Embarked"] = [avg_age if (str(x) == "nan") else x for x in data_frame["Embarked"]]

            data_frame = data_frame.drop("Ticket", axis=1)

            data_frame.to_csv(Memory.getPath() / "preprocessed/pptitanic.csv")

        data_frame = Memory.scale(data_frame)

        return data_frame, "Survived"

    @staticmethod
    def loadCancer():
        data_frame, preprocessed = Memory.load("cancer.csv")
        data_frame = Memory.scale(data_frame)

        return data_frame, "MEDV"

    @staticmethod
    def loadWine():
        return

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

    @staticmethod
    def scale(data_frame):
        x = data_frame.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        data_frame = pd.DataFrame(x_scaled, columns=data_frame.columns)

        return data_frame

