"""
Dynamic.py

Contains the DynamicModel class, which is a simple wrapper around a machine learning model (e.g. Random Forest) that can be trained on a specific train split of the data. 
The DynamicModelManager class manages multiple DynamicModels, each trained on a different train split of the data.

"""

import pandas as pd
import pickle as pkl
import numpy as np

from typing import Union, Dict, List, Tuple, Optional, Protocol
from sklearn.ensemble import RandomForestRegressor


class AnyModel(Protocol):
    def fit(self, X, y) -> "AnyModel": ...
    def predict(self, X) -> np.ndarray: ...


class DynamicModel:

    """
    Class for a Dynamic model that can be trained on a specific train split
    """

    def __init__(
            self, 
            step: int,
            train_split: Tuple[int, int],
            model: AnyModel = RandomForestRegressor()
        ):

        self.step = step
        self.train_split = train_split #this should be a tuple of month_ids
        self.model = model 

    def fit(
            self,
            data: pd.DataFrame, 
            features: List[str], 
            target: str,
        ):

        # TODO figure out way to deal with duplicate data

        # lead the target variable by the step size
        data["target_month_id"] = data["month_id"] + self.step
        data["target"] = data.groupby("country_id")[target].shift(-self.step)

        # filter the data to only include rows where the target_month_id is within the train split
        data = data[data['target_month_id'].between(*self.train_split)]

        # drop NAs
        data = data.dropna(subset=["target"])

        X = data[features]
        y = data[target]

        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:

        return self.model.predict(X)
    

class DynamicModelManager:

    """
    Class for managing multiple DynamicModels, each trained on a different train split of the data.
    """

    def __init__(
            self, 
            steps: List[int], 
            data: pd.DataFrame,
            features: List[str], 
            target: str,
            train_window_size: int, 
            test_window_size: int,
            slide_window_size: int,
            full_split: Tuple[int, int]
        ):
        
        self.data = data
        self.features = features
        self.target = target
        self.models = {}
        self.steps = steps
        self.train_window_size = train_window_size #should be in months
        self.test_window_size = test_window_size # should be in months
        self.slide_window_size = slide_window_size # should be in months
        self.full_split = full_split #should be month_id tuple

        self._is_fitted = False

    def fit(self):
    
        # set the first month of the first training split
        month_0 = self.full_split[0] + max(self.steps)
        # set the first month of the last training split
        month_max = self.full_split[1] - self.train_window_size

        # TODO calculate number of models to be fitted
        # TODO implement some sort of progress bar
        
        for step in self.steps:

            start_month = month_0
            end_month = month_0 + self.train_window_size

            while start_month <= month_max:

                # TODO iterate over train_split
                train_split = (start_month, end_month)

                # init model
                model = DynamicModel(
                    step, train_split
                )

                # fit model
                #model.fit(self.data, self.features, self.target)

                # save model
                self.models[f"{start_month}_{step}"] = model

                start_month += self.slide_window_size
                end_month += self.slide_window_size

        print(f"Finished fitting all models")

        self._is_fitted = True
        
        ...


    def save_artifact(self):
        ...


    def predict(self):

        if not self._is_fitted:
            print("ERROR") # TODO make usererror?

        ...