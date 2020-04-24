import time

from statistics import mean

import shap
from sklearn.feature_selection import VarianceThreshold, SelectPercentile
from sklearn.model_selection import cross_val_score

from metalfi.src.data.meta.importance.shap import ShapImportance


class MetaFeatureSelection:

    def __init__(self, meta_data, target_names):
        self.__X = meta_data.drop(target_names, axis=1)
        self.__Y = meta_data[target_names]
        self.__target_names = target_names
        self.__sets = {}

        support = VarianceThreshold(threshold=0.2).fit(self.__X).get_support(indices=True)
        features = [x for x in list(self.__X.columns) if list(self.__X.columns).index(x) in support]

        self.__X = self.__X[features]

        """cm = self.__X.corr()
        remove = []

        for i in range(len(cm.columns)):
            for j in range(i, len(cm.columns)):
                col = cm.columns[i]
                row = cm.index[j]
                if (cm.iloc[i, j] >= 0.8) and (col != row):
                    if not (col in remove):
                        if not (row in remove):
                            remove.append(col)
                            
        self.__X = self.__X.drop(remove, axis=1)"""

    def get_sets(self):
        return self.__sets

    def select(self, meta_model, scoring, k):
        for target in self.__target_names:
            y = self.__Y[target]

            percentiles = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50)
            p, _ = self.percentile_search(meta_model, scoring, y, percentiles, k)

            percentiles = (p - 4, p - 3, p - 2, p - 1, p, p + 1, p + 2, p + 3, p + 4)
            _, features = self.percentile_search(meta_model, scoring, y, percentiles, k)

            """meta_model.fit(self.__X[features], y)
            imp = shap.TreeExplainer(meta_model, self.__X[features]).shap_values(self.__X[features])
            shap.summary_plot(imp, self.__X[features], plot_type="bar")"""

            self.__sets[target] = features

    def percentile_search(self, meta_model, scoring, y, percentiles, k):
        results = []
        subsets = []

        for p in percentiles:
            support = SelectPercentile(score_func=scoring, percentile=p).fit(self.__X, y).get_support(indices=True)
            features = [x for x in list(self.__X.columns) if list(self.__X.columns).index(x) in support]

            X = self.__X[features]
            subsets.append(features)
            results.append(mean(cross_val_score(estimator=meta_model, X=X, y=y, cv=k)))

        index = results.index(max(results))
        p = percentiles[index]
        f = subsets[index]

        return p, f

    @staticmethod
    def metaFeatureImportance(meta_data, all_targets, models, targets, subsets):
        importance = {}
        all_X = meta_data.drop(all_targets, axis=1)
        Y = meta_data[targets]

        for target in targets:
            this_target = list()
            for model, name, category in models:
                X = all_X[subsets[name][target]]
                y = Y[target]
                s = ShapImportance(None)

                if category == "linear":
                    imp = s.linearShap(model, X, y)
                elif category == "tree":
                    imp = s.treeRegressionShap(model, X, y)
                else:
                    imp = s.kernelShap(model, X, y, 5)

                this_target.append(imp)

            importance[target] = this_target

        return importance
