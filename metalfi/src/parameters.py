from typing import List, Tuple

import numpy as np
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class Parameters:
    """
    Contains all experimental parameters as attributes.

    Attributes
    ----------
        metrics : (List[str])
            Evaluation metrics for meta-models.
        targets : (List[str])
            Names of all meta-targets.
        base_models : (List[Tuple[Estimator, str, str]])
            Contains base-models (classifiers), their respective names and categories.
        meta_models : (List[Tuple[Estimator, str, str]])
            Contains meta-models (regression models), their respective names and categories.
    """

    metrics = ["R^2"]
    targets = ["linSVC_SHAP", "LOG_SHAP", "RF_SHAP", "NB_SHAP", "SVC_SHAP", "DT_SHAP",
               "linSVC_LIME", "LOG_LIME", "RF_LIME", "NB_LIME", "SVC_LIME", "DT_LIME",
               "linSVC_PIMP", "LOG_PIMP", "RF_PIMP", "NB_PIMP", "SVC_PIMP", "DT_PIMP",
               "linSVC_LOFO", "LOG_LOFO", "RF_LOFO", "NB_LOFO", "SVC_LOFO", "DT_LOFO"]

    # Hyperparameters:
    #   (probability): Required for LIME-importance.
    #   (max_iter): Fix convergence issues.
    #   (random_state): Ensure reproducibility.
    base_models = [(RandomForestClassifier(random_state=115), "RF", "tree"),
                   (DecisionTreeClassifier(random_state=115), "DT", "tree"),
                   (SVC(probability=True, random_state=115), "SVC", "kernel"),
                   (LogisticRegression(max_iter=1000, random_state=115), "LOG", "linear"),
                   (SVC(kernel="linear", probability=True, random_state=115), "linSVC", "linear"),
                   (GaussianNB(), "NB", "kernel")]

    meta_models = [(RandomForestRegressor(random_state=115), "RF", "tree"),
                   (DecisionTreeRegressor(random_state=115), "DT", "tree"),
                   (SVR(), "SVR", "kernel"),
                   (LinearRegression(), "LIN", "linear"),
                   (LinearSVR(max_iter=10000, random_state=115), "linSVR", "linear")]

    @staticmethod
    def calculate_metrics(y_test: List[float], y_pred: List[float]) -> List[float]:
        """
        Compute different performance metrics for model predictions.

        Parameters
        ----------
            y_test : Target vector of the test data.
            y_pred : Predicted target vector of the test data.

        Returns
        -------
            Performance of prediction according to different metrics (currently: Coefficient of determination).

        """
        r_2 = 0 if np.std(y_test) <= 0.001 or np.std(y_pred) <= 0.001 else \
            r2_score(y_test, y_pred)

        return [r_2]

    @staticmethod
    def fi_measures() -> List[str]:
        """

        Returns
        -------
            List of all feature importance measures used to compute meta-targets.
        """
        return list(dict.fromkeys(map(lambda x: x[-4:], Parameters.targets)))

    @staticmethod
    def question_5_parameters() -> Tuple[List[str], List[str], List[str]]:
        """

        Returns
        -------
            Experimental parameters for research question 5.

        """
        return ["linSVR", "RF", "DT", "SVR", "LIN"], list(filter(lambda x: x.endswith("_PIMP"), Parameters.targets)), \
               ["LM", "FMF", "All", "Auto"]
