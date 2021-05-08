import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class Parameters:

    metrics = {0: "R^2", 1: "RMSE", 2: "r"}
    targets = ["linSVC_SHAP", "LOG_SHAP", "RF_SHAP", "NB_SHAP", "SVC_SHAP", "DT_SHAP",
               "linSVC_LIME", "LOG_LIME", "RF_LIME", "NB_LIME", "SVC_LIME", "DT_LIME",
               "linSVC_PIMP", "LOG_PIMP", "RF_PIMP", "NB_PIMP", "SVC_PIMP", "DT_PIMP",
               "linSVC_LOFO", "LOG_LOFO", "RF_LOFO", "NB_LOFO", "SVC_LOFO", "DT_LOFO"]

    base_models = [(RandomForestClassifier(oob_score=True, random_state=115), "RF", "tree"),
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
    def calculate_metrics(y_train, y_test, y_pred):
        r_2 = 1 - (sum([(y_pred[i] - y_test[i]) ** 2 for i in range(len(y_pred))]) /
                   sum([(np.mean(y_train) - y_test[i]) ** 2 for i in range(len(y_pred))]))
        rmse = np.sqrt(np.mean(([(y_pred[i] - y_test[i]) ** 2 for i in range(len(y_pred))])))
        base = np.sqrt(np.mean(([(np.mean(y_train) - y_test[i]) ** 2 for i in range(len(y_pred))])))
        r = np.corrcoef(y_pred, y_test)[0][1]

        return [r_2, rmse / base, r]

    @staticmethod
    def fi_measures():
        return list(dict.fromkeys(map(lambda x: x[-4:], Parameters.targets)))

    @staticmethod
    def question_5_parameters():
        return ["linSVR"], list(filter(lambda x: x.endswith("_SHAP"), Parameters.targets)), ["LM"]
