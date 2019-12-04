from metalfi.src.data.meta.importance.featureimportance import FeatureImportance
from rfpimp import *

class DropColumnImportance(FeatureImportance):

    def __init__(self, dataset):
        super(DropColumnImportance, self).__init__(dataset)

    def calculateScores(self):
        models = self._linear_models + self._kernel_models

        for model in models:
            self.dropcolImportance(model, self._target)

        for model in self._tree_models:
            self.oobDropcolImportance(model, self._target)

    def dropcolImportance(self, model, target):
        X = self._data_frame.drop(target, axis=1)
        y = self._data_frame[target]

        model.fit(X, y)
        imp = dropcol_importances(model, X, y)
        plot_importances(imp).view()

    def oobDropcolImportance(self, model, target):
        X = self._data_frame.drop(target, axis=1)
        y = self._data_frame[target]

        model.fit(X, y)
        imp = oob_dropcol_importances(model, X, y)
        plot_importances(imp).view()
