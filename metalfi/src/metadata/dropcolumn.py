from pandas import DataFrame
from rfpimp import *
from sklearn.preprocessing import StandardScaler

from metalfi.src.metadata.featureimportance import FeatureImportance


class DropColumnImportance(FeatureImportance):

    def __init__(self, dataset):
        super(DropColumnImportance, self).__init__(dataset)
        self._name = "_LOFO"

    def calculateScores(self):
        for type in self._models.keys():
            for model in self._models[type].values():
                if type == "tree":
                    self._feature_importances.append(self.oobDropcolImportance(model, self._target))
                else:
                    self._feature_importances.append(self.dropcolImportance(model, self._target))

    def dropcolImportance(self, model, target):
        sc = StandardScaler()
        X = DataFrame(data=sc.fit_transform(self._data_frame.drop(target, axis=1)),
                      columns=self._data_frame.drop(target, axis=1).columns)
        y = self._data_frame[target]

        model.fit(X, y)
        imp = dropcol_importances(model, X, y)
        return imp

    def oobDropcolImportance(self, model, target):
        sc = StandardScaler()
        X = DataFrame(data=sc.fit_transform(self._data_frame.drop(target, axis=1)),
                      columns=self._data_frame.drop(target, axis=1).columns)
        y = self._data_frame[target]

        model.fit(X, y)
        imp = oob_dropcol_importances(model, X, y)
        return imp
