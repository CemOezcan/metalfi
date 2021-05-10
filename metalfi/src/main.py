import time

from metalfi.src.controller import Controller
from metalfi.src.visualization import Visualization


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

        c.metaFeatureImportances()

        # Compare
        start = time.time()
        data.reverse()
        c.compare(data)
        end = time.time()
        print(end - start)

        # Tests
        Visualization.performance()
        Visualization.cleanUp()
        c.questions(data)
        Visualization.runtime_boxplot(100000000, ["LOFO", "PIMP"], ["landmarking", "univariate", "data"], "fast")
        Visualization.runtime_boxplot(100000000, ["LOFO"], ["multivariate"], "fast_multi")
        Visualization.runtime_boxplot(100000000, ["SHAP", "LIME"], ["total"], "slow")
        Visualization.runtime_graph("fast_graph")
        Visualization.createHistograms()
        Visualization.correlateTargets()
        Visualization.correlateMetrics()
        Visualization.metaFeatureImportance()


if __name__ == '__main__':
    Main().main()
