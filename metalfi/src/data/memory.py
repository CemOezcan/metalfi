import pickle

import numpy as np
import pandas as pd

from pathlib import Path
from pandas import DataFrame
from sklearn.datasets import load_wine, load_iris, load_boston, fetch_openml, load_diabetes
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, LabelEncoder


class Memory:

    def __init__(self):
        return

    @staticmethod
    def load(name, dir=None):
        path = Memory.getPath()
        if not (dir is None):
            return pd.read_csv(path / (dir + "/" + name))

        try:
            data = pd.read_csv(path / ("preprocessed/pp" + name)), True
        except FileNotFoundError:
            data = pd.read_csv(path / ("raw/" + name)), False

        return data

    @staticmethod
    def loadBoston():
        boston = load_boston()
        data_frame = DataFrame(data=boston.data, columns=boston['feature_names'])

        est = KBinsDiscretizer(n_bins=2, encode='ordinal')
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

    @staticmethod
    def loadOpenML():
        datasets = list()
        ids = [("tic-tac-toe", 1), ("phoneme", 1),
               ("banknote-authentication", 1), ("haberman", 1), ("servo", 1), ("cloud", 2),
               ("primary-tumor", 2), ("EgyptianSkulls", 1), ("SPECTF", 2), ("cpu", 2),
               ("bodyfat", 2), ("Engine1", 1), ("ESL", 2), ("ilpd-numeric", 2),
               ("credit-approval", 1), ("vowel", 3), ("socmob", 2), ("ERA", 1), ("LEV", 1), ("credit-g", 1), ("cmc", 2),
               ("bank8FM", 2), ("wind", 2)]

        for name, version in ids:
            dataset = fetch_openml(name=name, version=version, as_frame=True)
            categories = fetch_openml(name=name, version=version, as_frame=False)["categories"]
            all_features = dataset["feature_names"]
            cat_features = list()
            data_frame = dataset["frame"].dropna(axis=0)
            target = dataset["target_names"][0]

            X = data_frame.drop(target, axis=1)
            X_cat = X

            y = data_frame[target]

            sorted_categories = list()
            for feature in all_features:
                if str(data_frame[feature].dtypes) == "category":
                    cat_features.append(feature)
                    sorted_categories.append(categories[feature])
                else:
                    X_cat = X_cat.drop(feature, axis=1)

            if str(dataset["target"].dtypes) == "category":
                y_enc = LabelEncoder()
                y = y_enc.fit_transform(y)

            else:
                est = KBinsDiscretizer(n_bins=2, encode='ordinal')
                y = est.fit_transform(list(map(lambda x: [x], y)))

            X_num = X.drop(X_cat.columns, axis=1)
            num_features = list(set(all_features) - set(cat_features))

            X_enc = OrdinalEncoder(sorted_categories)
            X_cat = X_enc.fit_transform(X_cat)

            data_frame = DataFrame(data=np.c_[np.c_[X_cat, X_num], y],
                                   columns=cat_features + num_features + [target])

            datasets.append((data_frame, name, target))

        return datasets

    @staticmethod
    def storeMetaFeatures(data):
        path = Memory.getPath() / "features/selected"
        if not path.is_file():
            pickle.dump(data, open(path, 'wb'))

    @staticmethod
    def loadMetaFeatures():
        try:
            path = Memory.getPath() / "features/selected"
            file = open(path, 'rb')
            data = pickle.load(file)
            file.close()

        except FileNotFoundError:
            data = None

        return data

    @staticmethod
    def storeInput(data, name):
        path = Memory.getPath() / ("input/" + name + "meta.csv")
        if not path.is_file():
            data.to_csv(path, index=None, header=True)

    def storeOutput(self, data):
        return

    @staticmethod
    def storeModel(model, name, support):
        path = Memory.getPath() / ("model/" + name)
        if not path.is_file():
            file = open(path, 'wb')
            pickle.dump(model, file)
            file.close()

    @staticmethod
    def loadModel(names):
        models = list()
        for name in names:
            path = Memory.getPath() / ("model/" + name)
            file = open(path, 'rb')
            data = pickle.load(file)
            file.close()
            models.append((data, name))

        return models

    @staticmethod
    def renewModel(model, name):
        path = Memory.getPath() / ("model/" + name)
        file = open(path, "wb")
        pickle.dump(model, file)
        file.close()

    @staticmethod
    def storeDataFrame(data, name, directory):
        path = Memory.getPath() / ("output/" + directory + "/" + name + ".csv")
        if not path.is_file():
            data.to_csv(path, header=True)

    def storeVisual(self, data):
        return

    @staticmethod
    def getPath():
        path = Path(__file__).parents[2] / "data"
        return path
