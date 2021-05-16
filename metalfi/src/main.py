import math
import multiprocessing
import time

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR
from tqdm import tqdm

from metalfi.src.controller import Controller
from metalfi.src.memory import Memory
from metalfi.src.parameters import Parameters
from metalfi.src.visualization import Visualization
import openml


class Main(object):
    """
    Main Class
    """

    @staticmethod
    def main():
        """
        Main method. Computations start here.
        """
        # Calculate meta-datasets (if necessary)
        start = time.time()
        c = Controller()
        end = time.time()
        print(end - start)
        return

        # Train meta-models (if necessary)
        start = time.time()
        c.train_meta_models()
        end = time.time()
        print(end - start)

        data = Memory.get_contents("model")

        # Load trained meta-models from storage and get evaluation results
        start = time.time()
        c.estimate(data)
        end = time.time()
        print(end - start)

        c.meta_feature_importances()

        # Compare
        start = time.time()
        data.reverse()
        c.compare(data)
        end = time.time()
        print(end - start)

        # Tests
        Visualization.performance()
        Visualization.clean_up()
        c.questions(data)
        Visualization.runtime_boxplot(100000000, ["LOFO", "PIMP"], ["landmarking", "univariate", "data"], "fast")
        Visualization.runtime_boxplot(100000000, ["LOFO"], ["multivariate"], "fast_multi")
        Visualization.runtime_boxplot(100000000, ["SHAP", "LIME"], ["total"], "slow")
        Visualization.runtime_graph("fast_graph")
        Visualization.create_histograms()
        Visualization.correlate_targets()
        Visualization.correlate_metrics()
        Visualization.meta_feature_importance()


if __name__ == '__main__':
    Main().main()
