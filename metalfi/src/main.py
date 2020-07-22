import sys
import time

from metalfi.src.controller import Controller
from metalfi.src.visual.visualization import Visualization


class Main(object):

    @staticmethod
    def main():
        data = ["Titanic", "Iris", "Cancer", "Wine", "Boston", "tic-tac-toe", "banknote-authentication",
                "haberman", "servo", "cloud", "primary-tumor", "EgyptianSkulls", "SPECTF", "cpu", "bodyfat",
                "Engine1", "ESL", "ilpd-numeric", "credit-approval", "vowel", "socmob", "ERA", "LEV", "cmc", "credit-g",
                "phoneme", "bank8FM", "wind"]

        # Calculate meta-datasets (if necessary)
        start = time.time()
        c = Controller()
        end = time.time()
        print(end - start)

        # Train meta-models (if necessary)
        start = time.time()
        c.trainMetaModel()
        end = time.time()
        print(end - start)

        # Load trained meta-models from storage and get evaluation results
        start = time.time()
        c.evaluate(data)
        end = time.time()
        print(end - start)

        #c.metaFeatureImportances()

        # Compare
        start = time.time()
        c.compare(data)
        end = time.time()
        print(end - start)

        # Tests
        Visualization.cleanUp()
        c.questions(data, -4)
        Visualization.compareMeans("q2")
        Visualization.compareMeans("q3")
        Visualization.compareMeans("q4")
        Visualization.compareMeans("q5")
        Visualization.runtime_boxplot(100000000, ["LOFO", "PIMP"], ["landmarking", "univariate", "data"], "fast")
        Visualization.runtime_boxplot(100000000, ["SHAP", "LIME"], ["total"], "slow")
        Visualization.createHistograms()
        Visualization.correlateTargets()
        Visualization.correlateMetrics()
        Visualization.performance()
        Visualization.metaFeatureImportance()


if __name__ == '__main__':
    Main().main()
