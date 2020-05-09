import time

from statistics import mean

import shap
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, SelectFromModel
from sklearn.model_selection import cross_val_score

from metalfi.src.data.meta.importance.shap import ShapImportance


class MetaFeatureSelection:

    def __init__(self, meta_data, target_names):
        self.__X = meta_data.drop(target_names, axis=1)
        fmf = [x for x in self.__X.columns if "." not in x]
        self.__X = self.__X[fmf]
        self.__Y = meta_data[target_names]
        self.__target_names = target_names

        support = VarianceThreshold(threshold=0.2).fit(self.__X).get_support(indices=True)
        features = [x for x in list(self.__X.columns) if list(self.__X.columns).index(x) in support]

        self.__X = self.__X[features]

    def select(self, meta_model, scoring, percentiles=(5, 10, 15, 20, 25, 30), k=10, tree=False):
        sets = {}
        for target in self.__target_names:
            y = self.__Y[target]
            print("Select for target: " + target)
            if tree:
                sel = SelectFromModel(estimator=meta_model).fit(self.__X, y)
                support = sel.get_support(indices=True)
                features = [x for x in list(self.__X.columns) if list(self.__X.columns).index(x) in support]
            else:
                p, features = self.percentile_search(meta_model, scoring, y, percentiles, k, self.__X)

            sets[target] = features if (len(features) != 0) else list(self.__X.columns)

        return sets

    @staticmethod
    def percentile_search(meta_model, scoring, y, percentiles, k, new_X):
        results = []
        subsets = []

        for p in percentiles:
            support = SelectPercentile(score_func=scoring, percentile=p).fit(new_X, y).get_support(indices=True)
            features = [x for x in list(new_X.columns) if list(new_X.columns).index(x) in support]

            X = new_X[features]
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
