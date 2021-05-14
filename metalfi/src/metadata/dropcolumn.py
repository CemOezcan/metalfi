
from pandas import DataFrame
from rfpimp import *
from sklearn.preprocessing import StandardScaler

from metalfi.src.metadata.featureimportance import FeatureImportance


class DropColumnImportance(FeatureImportance):
    """
    LOFO-importance.
    """
    def __init__(self, dataset: 'Dataset'):
        super(DropColumnImportance, self).__init__(dataset)
        self._name = "_LOFO"

    def calculate_scores(self):
        for type in self._models.keys():
            for name in self._models[type].keys():
                model = self._models[type][name]
                if type == "tree" and name != "DT":
                    self._feature_importances.append(self.__oob_dropcol_importance(model, self._target))
                else:
                    self._feature_importances.append(self.__dropcol_importance(model, self._target))

    def __dropcol_importance(self, model, target: str) -> DataFrame:
        sc = StandardScaler()
        X = DataFrame(data=sc.fit_transform(self._data_frame.drop(target, axis=1)),
                      columns=self._data_frame.drop(target, axis=1).columns)
        y = self._data_frame[target]

        model.fit(X, y)
        imp = dropcol_importances(model, X, y)
        return imp

    def __oob_dropcol_importance(self, model, target: str) -> DataFrame:
        sc = StandardScaler()
        X = DataFrame(data=sc.fit_transform(self._data_frame.drop(target, axis=1)),
                      columns=self._data_frame.drop(target, axis=1).columns)
        y = self._data_frame[target]

        model.fit(X, y)
        imp = oob_dropcol_importances(model, X, y)
        return imp
