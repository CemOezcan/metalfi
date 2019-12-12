from metalfi.src.data.meta.importance.featureimportance import FeatureImportance
from rfpimp import *
from eli5.sklearn import PermutationImportance as eli5PI

import eli5


class PermutationImportance(FeatureImportance):

    def __init__(self, dataset):
        super(PermutationImportance, self).__init__(dataset)
        self._name = "_perm"

    def calculateScores(self):
        # TODO: Calc. importances for feature subsets that are multicollinear
        models = self._tree_models + self._linear_models + self._kernel_models

        for model in models:
            # self.cvPermutationImportance(model, self._target)
            self._feature_importances.append(self.permutationImportance(model, self._target))
            # self.eli5PermutationImportance(model, self._target)

    def cvPermutationImportance(self, model, target):
        # TODO: Calc. importances on testset?
        X = self._data_frame.drop(target, axis=1)
        y = self._data_frame[target]

        imp = cv_importances(model, X, y)
        plot_importances(imp).view()

    def permutationImportance(self, model, target):
        X = self._data_frame.drop(target, axis=1)
        y = self._data_frame[target]

        model.fit(X, y)
        imp = importances(model, X, y)
        plot_importances(imp).view()
        return imp

    def eli5PermutationImportance(self, model, target):
        X = self._data_frame.drop(target, axis=1)
        y = self._data_frame[target]

        model.fit(X, y)
        imp = eli5PI(model, random_state=101, cv="prefit").fit(X, y)

        print(eli5.format_as_text(eli5.explain_weights(imp)))
        print(X.columns)
