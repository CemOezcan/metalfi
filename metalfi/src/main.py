from metalfi.src.controller import Controller
from metalfi.src.memory import Memory
from metalfi.src.visualization import Visualization


class Main:
    """
    Main Class
    """

    @staticmethod
    def main():
        """
        Main method:
            - Calculate meta-datasets (if necessary)
            - Train meta-models (if necessary)
            - Load trained meta-models from storage (if necessary) and get evaluation results
            - Compute meta-feature importance estimates (if necessary)
            - Compare feature selection approaches (if necessary)
            - Employ tests and visualize results
        """
        c = Controller()
        c.train_meta_models()
        data = Memory.get_contents("model")
        c.estimate(data)
        c.meta_feature_importances()
        c.compare(data)
        c.questions(data)

        Visualization.performance()
        Visualization.clean_up()
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
