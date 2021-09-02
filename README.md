# Meta-Learning Feature Importance

This repository contains the code to reproduce the experiments of the paper

> Bach, Jakob, and Cem Özcan. "Meta-Learning Feature Importance"

This document describes the repo structure and the steps to reproduce the experiments.
Input data and results data of the experimental pipelines are also available [online](https://bwdatadiss.kit.edu/dataset/xxx).

## Repo Structure

    .
    ├── metalfi                 
        ├── data                                  # Data sets and plots.
            ├── base_datasets                     # Preprocessed base-data sets.
            ├── meta_datasets                     # Meta-data sets.
            ├── meta_models                       # Meta-models.
            ├── output                            # Experimental results. The content of this folder is used to generate plots, which are also saved in this directory.
                ├── meta_feature_importance       # Meta-feature importance.
                ├── meta_prediction_performance   # Meta-model performance, overall and comparing various experimental dimensions.
                ├── meta_computation_time         # Computation times of all meta-feature subsets and meta-targets for all base-data sets.
                ├── feature_selection_performance # Feature-selection performance of MetaLFI and competitors.
        ├── src                                   # Source code.
    ├── .gitignore
    ├── LICENSE
    ├── README.md
    └── requirements.txt

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
The results will be saved in subdirectories of `data/`.
This might take some time, but the pipeline has progress bars.
Meta-data calculation and meta-model training account for nearly the whole runtime.

If necessary, one can delete (and subsequently recompute) all meta-data sets or all trained meta-models by passing 
`delete_meta=True` or `delete_models=True` as keyword arguments: 

```bash
python -m metalfi.src.main delete_meta=True delete_models=True
```
