# Meta-Learning Feature Importance

This repository contains the code to reproduce the experiments of the paper

> Bach, Jakob, and Cem Özcan. "Meta-Learning Feature Importance"

This document describes the repo structure and the steps to reproduce the experiments.
Input data and results data of the experimental pipelines are also available [online](https://bwdatadiss.kit.edu/dataset/xxx).

## Repo Structure

    .
    ├── metalfi                 
        ├── data                        # Data sets and plots
            ├── features                # Data on meta-feature importance
            ├── input                   # Meta-data sets
            ├── model                   # Meta-models
            ├── output                  # .csv files of experimental results. The content of this folder is used to generate the plots.
                ├── groups              # 
                ├── importance          # Meta-feature importances
                ├── models              # .csv files for each research question (q2 - q5). This data is used to create the critical differences plots.
                ├── predictions         # Meta-model performances
                ├── runtime             # .csv files with the calculation times of each meta-data set 
                ├── selection           # Accuracy scores of base models using different feature selection methods (Part of research question 5)
                ├── targets             # 
            ├── preprocessed            # Preprocessed data sets
            ├── raw                     # Contains two small data sets
            
        ├── src                         # Source code
            ├── metadata                # Scripts that handle the meta-data
            ├── model                   # Contains scripts that train and evaluate meta-models
    ├── .gitignore                 
    ├── requirements.txt           
    ├── start.py                   
    ├── LICENSE
    └── README.md

## Setup

Before running scripts to reproduce the experiments, you need to set up an environment with all necessary dependencies.
Our code is implemented in Python (version 3.7).

### Option 1: `conda` Environment

If you use `conda`, you can install the right Python version into a new `conda` environment
and activate the environment as follows:

```bash
conda create --name <conda-env-name> python=3.7
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
python start.py
```

This will start the meta-data calculations, the meta-model training as well as the generation of the experimental results. 
The results will be saved in subdirectories of `metalfi/metalfi/data/`.
On an average laptop, the calculations require approximately 20 hours to finish.

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
