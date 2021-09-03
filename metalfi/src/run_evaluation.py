"""Run evaluation

Main evaluation pipeline, including:
- critical-difference amd mean plots comparing experimental dimensions
- distribution of meta-targets
- correlation analysis between meta-features and meta-targets
- distribution of meta-feature importance

Saves CSV files and plots.

Usage: python -m metalfi.src.run_evaluation --help
"""

import os

from metalfi.src.evaluation import Evaluation
from metalfi.src.Parameters import Parameters
from metalfi.src.visualization import Visualization


def run_evaluation():
    # Evaluate meta-model performance for different research questions:
    Evaluation([x for x in os.listdir(Parameters.meta_model_dir) if x != ".gitignore"]).questions()

    Visualization.create_histograms()
    Visualization.correlate_targets()
    Visualization.meta_feature_importance()


if __name__ == '__main__':
    print('Evaluation pipeline started.')
    run_evaluation()
    print('Evaluation pipeline executed successfully.')
