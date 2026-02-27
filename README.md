# Code for Master's Thesis

Repository containing the code and notebooks used for the analysis of Master's thesis. 


## Structure of repository

The repository is structured as follows:
- `data/`: Contains the datasets used for analysis. This includes both raw and processed data. Due to size, these are not included in the repository.
- `notebooks/`: Contains the notebooks used for data analysis, model training, and evaluation.
- `src/`: Contains the code used to conduct the analysis. Main components:
    - `dynamic.py`: code containing the `DynamicModelManager` class, which is used to train models in shifting train and test windows and evaluate performance.
    - `auxilaries.py`: Contains auxiliary functions used.
- `figs/`: Contains the figures generated from the analysis. Not included in the repository.
- `viewsforecasting/`: a clone of the VIEWS fatalities002 model, used to replicate some of the individual constituent models.