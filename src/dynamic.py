"""
Dynamic.py

Contains the DynamicModel class, which is a simple wrapper around a machine learning model (e.g. Random Forest) that can be trained on a specific train split of the data. 
The DynamicModelManager class manages multiple DynamicModels, each trained on a different train split of the data.

"""
import time

import pandas as pd
import pickle as pkl
import numpy as np

from typing import Union, Dict, List, Tuple, Optional, Protocol
from dataclasses import dataclass
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor


class AnyModel(Protocol):
    def fit(self, X, y) -> "AnyModel": ...
    def predict(self, X) -> np.ndarray: ...


@dataclass(frozen=True)
class TrainSplit:
    start_month: int
    end_month: int
    step: int

@dataclass(frozen=True)
class TestSplit:
    start_month: int
    end_month: int
    step: int

class Prediction:

    def __init__(
            self,
            train_split: TrainSplit,
            test_split: TestSplit,
            predictions: pd.DataFrame
        ):

        self.train_split = train_split
        self.test_split = test_split
        self.predictions = predictions


    def __repr__(self) -> str:
        return (
            f"Prediction({self.train_split}, "
            f"{self.test_split}, "
            f"predictions_shape={self.predictions.shape})"
        )

class DynamicModel:

    """
    Class for a Dynamic model that can be trained on a specific train split
    """

    def __init__(
            self, 
            train_split: TrainSplit,
            model: AnyModel = RandomForestRegressor()
        ):

        self.train_split = train_split
        self.model = model 
        
        self.features = []
        self.target = ""

        self._is_fitted = False


    def fit(
            self,
            data: pd.DataFrame, 
            features: List[str], 
            target: str,
        ) -> None:

        data = data.copy()

        self.features = features
        self.target = target

        # lead the target variable by the step size
        data["target_month_id"] = data["month_id"] + self.train_split.step
        data["target"] = data.groupby("country_id")[target].shift(-self.train_split.step)

        # filter the data to only include rows where the target_month_id is within the train split
        data = data[data['target_month_id'].between(self.train_split.start_month, self.train_split.end_month)]

        # drop NAs
        data = data.dropna(subset=["target"])

        X = data[self.features]
        y = data[self.target]

        self.model.fit(X, y)

        self._is_fitted = True


    def predict(
            self, 
            data: pd.DataFrame,
    ) -> np.ndarray:
        
        if not self._is_fitted:
            raise ValueError("Model must be fitted before predicting")

        return self.model.predict(data)
    

    def __repr__(self):
        fitted = getattr(self, "_is_fitted", False)
        return (
            f"DynamicModel({self.train_split},"
            f"fitted={fitted})"
        )


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
        
        self.data = data.copy()
        self.features = features
        self.target = target
        self.models = {}
        self.predictions = []
        self.steps = steps
        self.train_window_size = train_window_size #should be in months
        self.test_window_size = test_window_size # should be in months
        self.slide_window_size = slide_window_size # should be in months
        self.full_split = full_split #should be month_id tuple

        self._is_fitted = False

    def fit(self) -> None:

        train_splits = self.get_train_splits()

        start_time = time.time()
    
        for split in tqdm(train_splits, total = len(train_splits), desc="Fitting models"):
                
            # init model
            model = DynamicModel(
                train_split=split
            )

            # fit model
            model.fit(self.data, self.features, self.target)

            # save model
            self.models[(split.start_month, split.step)] = model

        # print elapsed time
        elapsed = time.time() - start_time
        print(f"Finished fitting all models in {elapsed:.2f} seconds")

        # set fitted flag to True
        self._is_fitted = True


    def save_artifact(self) -> None:

        with open("../data/model_artifacts/dynamic_model_manager.pkl", "wb") as f:
            pkl.dump(self, f)


    def predict(
            self, 
            data: pd.DataFrame,
            test_window_size: Optional[int] = None,
            slide_window_size: Optional[int] = None
        ) -> None:

        data = data.copy()

        if not self._is_fitted:
            raise ValueError("Models must be fitted before predicting")
        
        # set window sizes
        test_window_size = test_window_size or self.test_window_size
        slide_window_size = slide_window_size or self.slide_window_size

        total_iterations = sum(
            len(self.get_test_splits(self.models[m], test_window_size, slide_window_size))
            for m in self.models
        )
        
        with tqdm(total=total_iterations, desc="Predicting") as pbar:
            for model in self.models:

                model = self.models[model]

                test_splits = self.get_test_splits(model, test_window_size, slide_window_size)

                for test_split in test_splits:

                    # lead the target variable by the step size
                    data["target_month_id"] = data["month_id"] + test_split.step

                    # filter the data to only include rows where the month_id is within the test split
                    test_data = self.data[self.data['month_id'].between(test_split.start_month, test_split.end_month)]

                    # drop NA
                    test_data = test_data.dropna(subset=self.features)

                    X_test = test_data[self.features]

                    predictions = model.predict(X_test)

                    predictions_df = pd.DataFrame({
                        "target_month_id": test_data["target_month_id"],
                        "country_id": test_data["country_id"],
                        "prediction": predictions
                    })

                    prediction = Prediction(
                        train_split=model.train_split,
                        test_split=test_split,
                        predictions=predictions_df
                    )

                    self.predictions.append(prediction)

                    pbar.update(1)

    
    def get_train_splits(self) -> List[TrainSplit]:

        train_splits = []

        # set the first month of the first training split
        month_min = self.full_split[0] + max(self.steps)
        # set the first month of the last training split
        month_max = self.full_split[1] - self.train_window_size
        
        for step in self.steps:

            start_month = month_min
            end_month = month_min + self.train_window_size

            while start_month <= month_max:

                train_split = TrainSplit(start_month, end_month, step)

                start_month += self.slide_window_size
                end_month += self.slide_window_size

                train_splits.append(train_split)

        return train_splits
    

    def get_test_splits(
            self, 
            model: DynamicModel,
            test_window_size: int,
            slide_window_size: int
        ) -> List[TestSplit]:

        test_splits = []

        # set the first month of the first training split
        month_min = self.full_split[0] + max(self.steps)
        # set the first month of the last training split
        month_max = self.full_split[1]

        # get windows before the train split
        start_month = model.train_split.start_month - test_window_size - 1
        end_month = model.train_split.start_month - 1

        while start_month >= month_min:

            test_split = TestSplit(start_month, end_month, model.train_split.step)

            start_month -= slide_window_size
            end_month -= slide_window_size

            if not (test_split.end_month >= model.train_split.start_month and test_split.start_month <= model.train_split.end_month):
                test_splits.append(test_split)

        # get windows after the train split
        start_month = model.train_split.end_month + 1
        end_month = model.train_split.end_month + test_window_size + 1

        while end_month <= month_max:
            
            test_split = TestSplit(start_month, end_month, model.train_split.step)

            start_month += slide_window_size
            end_month += slide_window_size

            if not (test_split.end_month >= model.train_split.start_month and test_split.start_month <= model.train_split.end_month):
                test_splits.append(test_split)

        return test_splits

