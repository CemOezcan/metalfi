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
you will find a Python script `start.py`. The command `python start.py` will start MetaLFI.
The meta-data as well as the experimental results will be generated and saved in subdirectories of `metalfi\metalfi\data`.

## Plots and Tables used in my Bachelor Thesis
The generated plots can be found in the folder `metalfi\metalfi\data\visual`. 
The tables containing meta-model performances can be found in the folder `metalfi\metalfi\data\output\predictions`. 

#Folder Structure
    .
    ├── metalfi                 
        ├── data
            ├── features                # Contains data on meta-feature importance
            ├── input                   # Contains meta-data sets
            ├── model                   # Contains meta-models
            ├── output                  # Contains .csv files of experimental results that. The contents of this folder are used to generate the plots.
                ├── importance          # Contains meta-feature importances
                ├── predictions         # Contains .csv files with meta-model performances
                ├── questions           # Contains .csv files for each research question (q2 - q5). This data is used to create the critical differences plots.
                    ├── q2
                    ├── q3
                    ├── q4
                    ├── q5
                ├── runtime             # Contains .csv files with the calculation times of each meta-data set 
                ├── selection           # Contains .csv files accuracy scores of base models using different feature selection methods (Part of research question 5)
            ├── preprocessed            # Contains preprocessed data sets
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

