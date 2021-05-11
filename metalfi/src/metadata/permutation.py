from pandas import DataFrame
from rfpimp import *
from sklearn.preprocessing import StandardScaler

from metalfi.src.metadata.featureimportance import FeatureImportance


class PermutationImportance(FeatureImportance):

    def __init__(self, dataset):
        super(PermutationImportance, self).__init__(dataset)
        self._name = "_PIMP"

    def calculateScores(self):
        for model in self._all_models:
            self._feature_importances.append(self.permutationImportance(model, self._target))

    def permutationImportance(self, model, target):
        sc = StandardScaler()
        X = DataFrame(data=sc.fit_transform(self._data_frame.drop(target, axis=1)),
                      columns=self._data_frame.drop(target, axis=1).columns)
        y = self._data_frame[target]

        model.fit(X, y)
        np.random.seed(115)
        imp = importances(model, X, y)
        return imp
