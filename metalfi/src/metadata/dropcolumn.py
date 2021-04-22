from pandas import DataFrame
from rfpimp import *
from sklearn.preprocessing import StandardScaler

from metalfi.src.metadata.featureimportance import FeatureImportance


class DropColumnImportance(FeatureImportance):

    def __init__(self, dataset):
        super(DropColumnImportance, self).__init__(dataset)
        self._name = "_LOFO"

    def calculateScores(self):
        for model in self._linear_models:
            self._feature_importances.append(self.dropcolImportance(model, self._target))

        for model in self._tree_models:
            new_model = RandomForestClassifier(oob_score=True, n_estimators=100, random_state=115)
            self._feature_importances.append(self.oobDropcolImportance(new_model, self._target))

        for model in self._kernel_models:
            self._feature_importances.append(self.dropcolImportance(model, self._target))

    def dropcolImportance(self, model, target):
        sc = StandardScaler()
        X = DataFrame(data=sc.fit_transform(self._data_frame.drop(target, axis=1)),
                      columns=self._data_frame.drop(target, axis=1).columns)
        y = self._data_frame[target]

        model.fit(X, y)
        imp = dropcol_importances(model, X, y)
        #plot_importances(imp).view()
        return imp

    def oobDropcolImportance(self, model, target):
        X = self._data_frame.drop(target, axis=1)
        y = self._data_frame[target]

        model.fit(X, y)
        imp = oob_dropcol_importances(model, X, y)
        #plot_importances(imp).view()
        return imp
