from metalfi.src.data.meta.importance.featureimportance import FeatureImportance
from rfpimp import *
from sklearn import linear_model
from sklearn.svm import SVC
from eli5.sklearn import PermutationImportance as eli5PI

import eli5


class PermutationImportance(FeatureImportance):

    def __init__(self, dataset):
        super(PermutationImportance, self).__init__(dataset)

    def calculateScores(self):
        reg = linear_model.LinearRegression()
        rf = RandomForestClassifier(random_state=101)
        svc = SVC(gamma='auto')

        target = self._dataset.getTarget()

        # TODO: Calc. importances for feature subsets that are multicollinear
        self.cvPermutationImportance(reg, target)
        self.permutationImportance(reg, target)
        self.eli5PermutationImportance(reg, target)

        self.cvPermutationImportance(rf, target)
        self.permutationImportance(rf, target)
        self.eli5PermutationImportance(rf, target)

        self.cvPermutationImportance(svc, target)
        self.permutationImportance(svc, target)
        self.eli5PermutationImportance(svc, target)

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

    def eli5PermutationImportance(self, model, target):
        X = self._data_frame.drop(target, axis=1)
        y = self._data_frame[target]

        model.fit(X, y)
        imp = eli5PI(model, random_state=101, cv="prefit").fit(X, y)

        print(eli5.format_as_text(eli5.explain_weights(imp)))
        print(X.columns)
