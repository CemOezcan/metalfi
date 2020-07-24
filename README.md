# metaLFI
Meta-Learning Feature Importance

# Getting Started
Clone this repository to your local computer and install the required packages on your virtual environment using the 
`requirements.txt` file as follows: 

 `pip install -r \Path\to\requirements.txt`
 
Python version: 
 
 `Python 3.8.1`
 
# How to Reproduce the Experimental Results
Open the terminal and navigate to the project's directory `\Path\to\metalfi`. In this directory, 
you will find a Python script `start.py`. Make sure that you are using the 
virtual environment, on which the required packages have been installed, and enter the command `python start.py`. 
This will start the meta-data calculations, the meta-model training as well as the generation of the experimental results. 
The results will be saved in subdirectories of `metalfi\metalfi\data`.

## Plots and Tables used in my Bachelor Thesis
The generated plots and tables can be found in the folder `metalfi\metalfi\data\visual`. 
The following Table maps figures in my bachelor thesis to the file names of their corresponding `.png` files in folder `visual`:  

| Figure number | 6.2 | 6.3 | 6.4 | 6.5 | 6.6 | 6.7 | 6.8 | 6.9 | 6.10 | 6.11 | 6.12 | 6.13 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| File name | R^2NON | R^2LIN | RF_SHAP | Histograms | targets_R^2 | base_R^2 | r | linSVR x LM | fast | fast_multi | slow | fast_graph |

The folder `visual` also contains `.csv` files that can be found as tables in my bachelor thesis: 
`Table 6.1` in the thesis corresponds to `visual\target_corr.csv`. The claim that our evaluation metrics are correlated, 
 is supported by `visual\metrics_corr.csv`.  
 

The tables containing meta-model performances can be found in the folder `metalfi\metalfi\data\output\predictions`. 
The `.csv` files in the `predictions` are named after their metric and their meta-target. 
For instance, the file that contains the performances, measured with R^2, 
of all meta-models predicting SHAP-based meta-targets is `predictions\r2xSHAP.csv`.

# Folder Structure

    .
    ├── metalfi                 
        ├── data                        # Data sets and plots
            ├── features                # Data on meta-feature importance
            ├── input                   # Meta-data sets
            ├── model                   # Meta-models
            ├── output                  # .csv files of experimental results. The content of this folder is used to generate the plots.
                ├── importance          # Meta-feature importances
                ├── predictions         # Meta-model performances
                ├── questions           # .csv files for each research question (q2 - q5). This data is used to create the critical differences plots.
                    ├── q2
                    ├── q3
                    ├── q4
                    ├── q5
                ├── runtime             # .csv files with the calculation times of each meta-data set 
                ├── selection           # Accuracy scores of base models using different feature selection methods (Part of research question 5)
            ├── preprocessed            # Preprocessed data sets
            ├── raw                     # Contains two small data sets
            ├── visual                  # Contains all plots
            
        ├── src                         # Source code
            ├── data                    # Scripts that handle the data
                ├── meta                # Meta-data related scripts
                    ├── importance      # Meta-target related scripts
            ├── model                   # Contains scripts that train and evaluate meta-models
            ├── visual                  # Scripts that visualize the generated data
    ├── .gitignore                 
    ├── requirements.txt           
    ├── start.py                   
    ├── LICENSE
    └── README.md

