"""Run experiments

Main experimental pipeline, including:
- download base datasets
- calculate meta-datasets
- train and evaluate meta-models
- determine meta-feature importance
- aggregate evaluation metrics, create tables and some plots

Usage: python -m metalfi.src.run_experiments --help
"""
import argparse
import os

from metalfi.src.controller import Controller
from metalfi.src.memory import Memory
from metalfi.src.parameters import Parameters


# "delete_inputs": Whether to delete base datasets and meta-datasets.
# "delete_outputs": Whether to delete meta-models and results.
def run_experiments(delete_inputs: bool = False, delete_outputs: bool = False):
    if delete_inputs:
        Memory.clear_directories([Parameters.base_dataset_dir, Parameters.meta_dataset_dir,
                                  Parameters.output_dir + "meta_computation_time"])
    if delete_outputs:
        directories = ["meta_feature_importance", "meta_prediction_performance", "feature_selection_performance"]
        directories = [Parameters.output_dir + x for x in directories]
        directories.append(Parameters.meta_model_dir)
        Memory.clear_directories(directories)

    c = Controller()  # download base datasets, creates meta-datasets
    c.train_meta_models()
    c.meta_feature_importances()
    c.estimate([x for x in os.listdir(Parameters.meta_model_dir) if x != ".gitignore"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs the experimental pipeline.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--delete_inputs', type=bool, default=False,
                        help='Delete all base datasets and meta-dataset from previous runs first.')
    parser.add_argument('--delete_outputs', type=bool, default=False,
                        help='Delete all meta-models and results from previous runs first.')
    print('Experimental pipeline started.')
    run_experiments(**vars(parser.parse_args()))
    print('Experimental pipeline executed successfully.')
