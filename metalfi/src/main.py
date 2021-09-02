import warnings
warnings.filterwarnings("ignore", message="IPython could not be loaded!")
import sys


from metalfi.src.controller import Controller
from metalfi.src.memory import Memory
from metalfi.src.visualization import Visualization


class Main:
    """
    Main Class
    """

    @staticmethod
    def main(delete_meta=False, delete_models=False):
        """
        Main method:
            - Calculate meta-datasets (if necessary)
            - Train meta-models (if necessary)
            - Load trained meta-models from storage (if necessary) and get evaluation results
            - Compute meta-feature importance estimates (if necessary)
            - Compare feature selection approaches (if necessary)
            - Employ tests and visualize results

        Parameters
        ----------
            delete_meta : (bool)
                Whether to delete meta-data.
            delete_models: (bool)
                Whether to delete meta-models.
        """
        directories = ["model", "output/importance", "output/predictions", "output/selection", "output/tables"]
        if delete_meta:
            Memory.clear_directory(["base_datasets", "meta_datasets", "output/runtime"] + directories)
        if delete_models:
            Memory.clear_directory(directories)

        c = Controller()
        c.train_meta_models()
        data = Memory.get_contents("model")
        c.estimate(data)
        c.meta_feature_importances()
        c.questions(data)

        Visualization.create_histograms()
        Visualization.correlate_targets()
        Visualization.meta_feature_importance()


if __name__ == '__main__':
    args = sys.argv[1:]
    fst = "True" in [x[12:] for x in args if "delete_meta=" in x]
    snd = "True" in [x[14:] for x in args if "delete_models=" in x]
    Main().main(delete_meta=fst, delete_models=snd)
