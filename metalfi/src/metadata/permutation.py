
from pandas import DataFrame
from rfpimp import *
from sklearn.preprocessing import StandardScaler

from metalfi.src.metadata.featureimportance import FeatureImportance


class PermutationImportance(FeatureImportance):
    """
    PIMP-importance.
    """
    def __init__(self, dataset: 'Dataset'):
        super(PermutationImportance, self).__init__(dataset)
        self._name = "_PIMP"

    def calculate_scores(self):
        for model in self._all_models:
            self._feature_importances.append(self.__permutation_importance(model, self._target))

    def __permutation_importance(self, model, target: str) -> DataFrame:
        sc = StandardScaler()
        X = DataFrame(data=sc.fit_transform(self._data_frame.drop(target, axis=1)),
                      columns=self._data_frame.drop(target, axis=1).columns)
        y = self._data_frame[target]

        model.fit(X, y)
        np.random.seed(115)
        imp = importances(model, X, y)
        return imp
