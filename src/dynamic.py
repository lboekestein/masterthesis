"""
Dynamic.py

Contains the DynamicModel class, which is a simple wrapper around a machine learning model (e.g. Random Forest) that can be trained on a specific train split of the data. 
The DynamicModelManager class manages multiple DynamicModels, each trained on a different train split of the data.

"""

import pandas as pd
import pickle as pkl
import numpy as np

from typing import Union, Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor


class DynamicModel:

    """
    Class for a Dynamic model that can be trained on a specific train split
    """

    def __init__(
            self, 
            data: pd.DataFrame, 
            features: List[str], 
            target: str,
            step: int,
            train_split: Tuple[int, int],
        ):

        self.data = data
        self.features = features
        self.target = target
        self.step = step
        self.train_split = train_split #this should be a tuple of month_ids
        self.model = RandomForestRegressor()

    def fit(self):

        # lead the target variable by the step size
        self.data["target_month_id"] = self.data["month_id"] + self.step
        self.data["target"] = self.data.groupby("country_id")[self.target].shift(-self.step)

        # filter the data to only include rows where the target_month_id is within the train split
        self.data = self.data[self.data['target_month_id'].between(*self.train_split)]

        # drop NAs
        self.data = self.data.dropna(subset=["target"])

        X = self.data[self.features]
        y = self.data[self.target]
        
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
            train_window_size: int, 
            test_window_size: int,
            full_split: Tuple[int, int]
        ):

        self.models = {}
        self.steps = steps
        self.train_window_size = train_window_size
        self.test_window_size = test_window_size
