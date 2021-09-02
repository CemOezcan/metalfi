# Meta-Learning Feature Importance

This repository contains the code to reproduce the experiments of the paper

> Bach, Jakob, and Cem Özcan. "Meta-Learning Feature Importance"

This document describes the repo structure and the steps to reproduce the experiments.
Input data and results data of the experimental pipelines are also available [online](https://bwdatadiss.kit.edu/dataset/xxx).

## Repo Structure

    .
    ├── metalfi                 
        ├── data                        # Data sets and plots
            ├── base_datasets           # Preprocessed base-data sets.
            ├── meta_datasets           # Meta-data sets
            ├── meta_models             # Meta-models
            ├── output                  # .csv files of experimental results. The content of this folder is used to generate the plots, which are also saved in this directory.
                ├── groups              # Group meta-model performance estimates by meta-feature subsets: 
                ├                            Box plots and critical differences diagrams visualize and compare 
                ├                            linear and non-linear meta-model performances based on the meta-feature 
                ├                            subsets, they were trained on.
                ├── importance          # SHAP-based meta-feature importance estimates: 
                ├                            Bar plots visualize the importance values of the top 15 meta-features. 
                ├── models              # Group meta-model performance estimates by meta-model types: 
                ├                            Box plots and critical differences diagrams visualize and compare meta-model 
                ├                            performance estimates, grouped by their respective regression model.
                ├── predictions         # All meta-model performance estimates over all meta-model configurations 
                ├                            and cross validation splits. 
                ├── runtime             # Computation times of all meta-feature subsets and meta-targets, 
                ├                            for all base-data sets. 
                ├── selection           # Group base-model accuracy scores by different feature selection approaches: 
                ├                            Bar plots and critical differences diagrams visualize and compare base-model 
                ├                            accuracies grouped by feature selection approaches. 
                ├── targets             # Group meta-model performance estimates by meta-targets:
                ├                            Box plots and critical differences diagrams visualize and compare meta-model 
                ├                            performance estimates grouped by meta-targets. 
        ├── src                         # Source code
    ├── .gitignore                 
    ├── requirements.txt           
    ├── start.py                   
    ├── LICENSE
    └── README.md

## Setup

Before running scripts to reproduce the experiments, you need to set up an environment with all necessary dependencies.
Our code is implemented in Python (version 3.8).

### Option 1: `conda` Environment

If you use `conda`, you can install the right Python version into a new `conda` environment
and activate the environment as follows:

```bash
conda create --name <conda-env-name> python=3.8
conda activate <conda-env-name>
```

### Option 2: `virtualenv` Environment

We used [`virtualenv`](https://virtualenv.pypa.io/) (version 20.4.0) to create an environment for our experiments.
First, make sure you have the right Python version available.
Next, you can install `virtualenv` with

```bash
python -m pip install virtualenv==20.4.0
```

To set up an environment with `virtualenv`, run


```bash
python -m virtualenv -p <path/to/right/python/executable> <path/to/env/destination>
```

Activate the environment in Linux with

```bash
source <path/to/env/destination>/bin/activate
```

Activate the environment in Windows (note the back-slashes) with

```cmd
<path\to\env\destination>\Scripts\activate
```

### Dependency Management

After activating the environment, you can use `python` and `pip` as usual.
To install all necessary dependencies for this repo, simply run

```bash
python -m pip install -r requirements.txt
```

If you make changes to the environment and you want to persist them, run

```bash
python -m pip freeze > requirements.txt
```

To leave the environment, run

```bash
deactivate
```

### Optional Dependencies

To use the environment in the IDE `Spyder`, you need to install `spyder-kernels` into the environment.

To run or create notebooks from the environment, you need to install `juypter` into the environment.
Next, you need to install a kernel for the environment:

```bash
ipython kernel install --user --name=<kernel-name>
```

After that, you should see (and be able to select) the kernel when running `Juypter Notebook` with

```bash
jupyter notebook
```

## Reproducing the Experiments

After setting up and activating an environment, you are ready to run the code.
From the top-level of this repo, run

```bash
python -m metalfi.src.main
```

This will start the meta-data calculations, the meta-model training as well as the generation of the experimental results. 
The results will be saved in subdirectories of `metalfi/metalfi/data/`.
On an average laptop, the calculations require approximately 20 hours to finish.

If necessary, one can delete (and subsequently recompute) all meta-data sets or all trained meta-models by passing 
`delete_meta=True` or `delete_models=True` as keyword arguments: 

```bash
python -m metalfi.src.main delete_meta=True delete_models=True
```

### Plots and Tables

The generated plots and tables can be found in the folder `metalfi/metalfi/data/visual/`. 
The following table maps figures in the article to the file names of their corresponding `.png` files in folder `visual`:  

| Figure number | 6.2 | 6.3 | 6.4 | 6.5 | 6.6 | 6.7 | 6.8 | 6.9 | 6.10 | 6.11 | 6.12 | 6.13 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| File name | R^2NON | R^2LIN | RF_SHAP | Histograms | targets_R^2 | base_R^2 | r | linSVR x LM | fast | fast_multi | slow | fast_graph |

The folder `visual` also contains `.csv` files that can be found as tables in the article: 
`Table 6.1` in the article corresponds to `visual/target_corr.csv`.
The claim that our evaluation metrics are correlated, is supported by `visual/metrics_corr.csv`.  

The tables containing meta-model performances can be found in the folder `metalfi/metalfi/data/output/predictions/`. 
The `.csv` files in the `predictions` are named after their metric and their meta-target. 
For instance, the file that contains the performances, measured with R^2,  of all meta-models predicting SHAP-based meta-targets is `predictions/r2xSHAP.csv`.
