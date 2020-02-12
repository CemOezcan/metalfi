from rfpimp import *

from metalfi.src.data.meta.importance.featureimportance import FeatureImportance


class PermutationImportance(FeatureImportance):

    def __init__(self, dataset):
        super(PermutationImportance, self).__init__(dataset)
        self._name = "_perm"

    def calculateScores(self):
        # TODO: Calc. importances for feature subsets that are multicollinear
        models = self._linear_models + self._tree_models + self._kernel_models

        for model in models:
            # self._feature_importances.append(self.cvPermutationImportance(model, self._target))
            self._feature_importances.append(self.permutationImportance(model, self._target))
            # self.eli5PermutationImportance(model, self._target)

    def cvPermutationImportance(self, model, target):
        # TODO: Calc. importances on testset?
        X = self._data_frame.drop(target, axis=1)
        y = self._data_frame[target]

        imp = cv_importances(model, X, y)
        plot_importances(imp).view()

        return imp

    def permutationImportance(self, model, target):
        X = self._data_frame.drop(target, axis=1)
        y = self._data_frame[target]

        #print(cross_val_score(model, X, y, cv=5))
        model.fit(X, y)
        imp = importances(model, X, y)
        plot_importances(imp).view()
        return imp
