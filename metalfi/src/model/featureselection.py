import os
import sys
import warnings
from multiprocessing.pool import Pool

import numpy as np

from statistics import mean
from sklearn.feature_selection import VarianceThreshold, SelectPercentile
from sklearn.model_selection import cross_val_score
import tqdm

from metalfi.src.metadata.shap import ShapImportance


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
            if tree:
                p, features = self.percentile_search(meta_model, scoring, y, [75], k, self.__X)
            else:
                p, features = self.percentile_search(meta_model, scoring, y, percentiles, k, self.__X)

            sets[target] = features if (len(features) != 0) else list(self.__X.columns)

        return sets

    @staticmethod
    def percentile_search(meta_model, scoring, y, percentiles, k, new_X):
        results = []
        subsets = []

        if len(percentiles) == 1:
            support = SelectPercentile(score_func=scoring, percentile=percentiles[0]).fit(new_X, y).get_support(indices=True)
            features = [x for x in list(new_X.columns) if list(new_X.columns).index(x) in support]

            subsets.append(features)
            p = percentiles[0]
            f = subsets[0]

            return p, f

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

        iterable = list()
        for target in targets:
            iterable += [(target, model, all_X[subsets[name][target]], Y[target], category)
                         for model, name, category in models]

        with Pool(processes=4) as pool:
            progress_bar = tqdm.tqdm(total=len(iterable), desc="Computing meta-feature importance")

            def update(param):
                progress_bar.update(n=1)

            results = [pool.map_async(MetaFeatureSelection.parallel_meta_importance, (x,), callback=update)
                       for x in iterable]
            results = [x.get()[0] for x in results]
            pool.close()
            pool.join()

        progress_bar.close()
        for target in targets:
            importance[target] = list()

        for target, imp in results:
            importance[target].append(imp)

        return importance

    @staticmethod
    def parallel_meta_importance(iterable):
        target, model, X, y, category = iterable
        s = ShapImportance(None)
        warnings.simplefilter("ignore")
        with open(os.devnull, 'w') as file:
            sys.stderr = file

        if category == "linear":
            imp = s.linearShap(model, X, y)
        elif category == "tree":
            imp = s.treeRegressionShap(model, X, y)
        else:
            imp = s.kernelShap(model, X, y, 5)

        array = imp["Importances"].values
        array = list(np.interp(array, (array.min(), array.max()), (0, 1)))
        sys.stderr = sys.__stderr__
        warnings.simplefilter("default")

        for i in range(len(imp.index)):
            imp.iloc[i, 0] = array[i]

        return target, imp
