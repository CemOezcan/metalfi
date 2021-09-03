import warnings
warnings.filterwarnings("ignore", message="IPython could not be loaded!")

import argparse
import os
import sys

from metalfi.src.controller import Controller
from metalfi.src.memory import Memory
from metalfi.src.parameters import Parameters
from metalfi.src.visualization import Visualization


class Main:
    """
    Main Class
    """

    @staticmethod
    def main(delete_inputs: bool = False, delete_outputs: bool = False):
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
            delete_inputs : (bool)
                Whether to delete base datasets and meta-datasets.
            delete_outputs: (bool)
                Whether to delete meta-models and results.
        """
        if delete_inputs:
            Memory.clear_directories([Parameters.base_dataset_dir, Parameters.meta_dataset_dir,
                                      Parameters.output_dir + "meta_computation_time"])
        if delete_outputs:
            directories = ["meta_feature_importance", "meta_prediction_performance", "feature_selection_performance"]
            directories = [Parameters.output_dir + x for x in directories]
            directories.append(Parameters.meta_model_dir)
            Memory.clear_directories(directories)

        c = Controller()
        c.train_meta_models()
        data = [x for x in os.listdir(Parameters.meta_model_dir) if x != ".gitignore"]
        c.estimate(data)
        c.meta_feature_importances()
        c.questions(data)

        Visualization.create_histograms()
        Visualization.correlate_targets()
        Visualization.meta_feature_importance()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs the experimental pipeline.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--delete_inputs', type=bool, default=False,
                        help='Delete all base datasets and meta-dataset from previous runs first.')
    parser.add_argument('--delete_outputs', type=bool, default=False,
                        help='Delete all meta-models and results from previous runs first.')
    print('Experimental pipeline started.')
    Main().main(**vars(parser.parse_args()))
    print('Experimental pipeline executed successfully.')
