import multiprocessing as mp
import os
from statistics import mean
import sys
from typing import Dict, List, Tuple
import warnings

import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.feature_selection import VarianceThreshold, SelectPercentile
from sklearn.model_selection import cross_val_score
import tqdm

from metalfi.src.metadata.shap import ShapImportance
from metalfi.src.parameters import Parameters


class MetaFeatureSelection:
    """
    Methods for feature importance estimation and selection.
    """

    def __init__(self, meta_data: DataFrame):
        self.__X = meta_data.drop(Parameters.targets, axis=1)
        fmf = [x for x in self.__X.columns if "." not in x]
        self.__X = self.__X[fmf]
        self.__Y = meta_data[Parameters.targets]

        support = VarianceThreshold(threshold=0.2).fit(self.__X).get_support(indices=True)
        features = [x for x in list(self.__X.columns) if list(self.__X.columns).index(x) in support]

        self.__X = self.__X[features]

    def select(self, meta_model, scoring, percentiles=(5, 10, 15, 20, 25, 30), k=10, tree=False) \
            -> Dict[str, Dict[str, List[str]]]:
        """
        Meta-feature selection, given a meta-model.
        Rank meta-features according to their `scoring` value and search for the percentile of al meta-features,
        that yields the best results.

        Parameters
        ----------
            meta_model : (scikit-learn model) The meta-model.
            scoring : (scikit-learn scoring function) Measure of feature importance.
            percentiles : (Tuple[int]) Search space.
            k : (int) k-fold cross validation is used to estimate model performances.
            tree : (bool) Whether the meta-model is a decision tree based model.

        Returns
        -------
            Optimal meta-feature subset.

        """
        sets = {}
        for target in Parameters.targets:
            y = self.__Y[target]
            if tree:
                _, features = self.__percentile_search(meta_model, scoring, y, [75], k, self.__X)
            else:
                _, features = self.__percentile_search(meta_model, scoring, y, percentiles, k, self.__X)

            sets[target] = features if (len(features) != 0) else list(self.__X.columns)

        return sets

    @staticmethod
    def __percentile_search(meta_model, scoring, y, percentiles, k, new_X):
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
    def meta_feature_importance(meta_data: DataFrame, models: List[Tuple[BaseEstimator, str, str]], targets: List[str],
                                subsets: Dict[str, Dict[str, List[str]]]) -> Dict[str, List[DataFrame]]:
        """
        Estimate model based meta-feature importance.

        Parameters
        ----------
            meta_data : Meta-data set.
            models : Meta-models.
            targets : Meta-targets.
            subsets : Subset of meta-features, whose importance is supposed to be estimated.

        Returns
        -------
            Meta-feature importance estimates for the given meta-features.
        """
        importance = {}
        all_targets = Parameters.targets
        all_X = meta_data.drop(all_targets, axis=1)
        Y = meta_data[targets]

        iterable = list()
        for target in targets:
            importance[target] = list()
            iterable += [(target, model, all_X[subsets[name][target]], Y[target], category)
                         for model, name, category in models]

        with mp.Pool(processes=mp.cpu_count() - 1) as pool:
            progress_bar = tqdm.tqdm(total=len(iterable), desc="Computing meta-feature importance")
            results = [
                pool.map_async(
                    MetaFeatureSelection.parallel_meta_importance,
                    (x,),
                    callback=(lambda x: progress_bar.update(n=1)))
                for x in iterable]

            results = [x.get()[0] for x in results]
            pool.close()
            pool.join()

        progress_bar.close()
        for target, imp in results:
            importance[target].append(imp)

        return importance

    @staticmethod
    def parallel_meta_importance(iterable: Tuple[str, BaseEstimator, DataFrame, DataFrame, str]) \
            -> Tuple[str, DataFrame]:
        """
        Compute SHAP-importance.

        Parameters
        ----------
            iterable : Tuple, contains meta-target name, meta-model, meta-data set, meta-target, meta-model category

        Returns
        -------
            SHAP-importance values
        """
        warnings.simplefilter("ignore")
        with open(os.devnull, 'w') as file:
            sys.stderr = file

        target, model, X, y, category = iterable
        s = ShapImportance(None)

        if category == "linear":
            imp = s.linear_shap(model, X, y)
        elif category == "tree":
            imp = s.tree_regression_shap(model, X, y)
        else:
            imp = s.kernel_shap(model, X, y)

        array = imp["Importances"].values
        array = list(np.interp(array, (array.min(), array.max()), (0, 1)))

        for i in range(len(imp.index)):
            imp.iloc[i, 0] = array[i]

        sys.stderr = sys.__stderr__
        warnings.simplefilter("default")

        return target, imp
