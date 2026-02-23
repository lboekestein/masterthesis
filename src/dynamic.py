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
    """
    Protocol for any machine learning model that implements fit and predict methods.
    """
    def fit(self, X, y) -> "AnyModel": 
        ...
    def predict(self, X) -> np.ndarray: 
        ...


@dataclass(frozen=True)
class TrainSplit:
    """
    Dataclass for a training split of the data.

    Attributes:
        start_month (int): the starting month of the training split (inclusive)
        end_month (int): the ending month of the training split (inclusive)
        step (int): the number of months the model is predicting into the future
    """
    start_month: int
    end_month: int
    step: int


@dataclass(frozen=True)
class TestSplit:
    """
    Dataclass for a test split of the data.

    Attributes:
        start_month (int): the starting month of the training split (inclusive)
        end_month (int): the ending month of the training split (inclusive)
        step (int): the number of months the model is predicting into the future
    """
    start_month: int
    end_month: int
    step: int


class Prediction:
    """
    Class for storing predictions from a DynamicModel on a specific test split.

    Attributes:
        train_split (TrainSplit): the training split the model was trained on
        test_split (TestSplit): the test split the predictions are for
        predictions (pd.DataFrame): a dataframe containing the predictions, with columns 'target_month_id', 'country_id', and 'prediction'
        prediction_col (str): the name of the column in predictions that contains the predicted values
        prediction_target (str): the name of the target variable that the model is predicting (e.g. 'ucdp_ged_sb_best_sum')
    """

    def __init__(
            self,
            train_split: TrainSplit,
            test_split: TestSplit,
            predictions: pd.DataFrame,
            prediction_col: str = "prediction",
            prediction_target: str = "ucdp_ged_sb_best_sum"
        ):

        self.train_split = train_split
        self.test_split = test_split
        self.predictions = predictions
        self.prediction_col = prediction_col
        self.prediction_target = prediction_target

        self.distance_ = self.calculate_distance()

    
    def calculate_distance(self) -> float:
        """
        Calculate the distance between the train and test splits in months.
        Distance is defined as the number of months between the closest edge of the train split to the middle of the test split. 
        If the train split overlaps with the test split, distance is not defined and an error is raised.
        
        Returns:
            float: the distance between the train and test splits in months
        """

        # compute the middle of the test split
        test_middle = (self.test_split.start_month + self.test_split.end_month) / 2
        
        # if the train split is after the test split, return the distance from the start of the train split to middle of the test split
        if self.train_split.start_month >= test_middle:
            return self.train_split.start_month - test_middle
        # if the train split is before the test split, return the distance from the end of the train split to middle of the test split
        elif self.train_split.end_month <= test_middle:
            return test_middle - self.train_split.end_month
        # if the train split overlaps with the test split, raise error
        else:
            raise ValueError("Train and test splits overlap, distance is not defined")


    def msle(
            self, 
            actuals: pd.DataFrame, 
        ) -> float:
        """
        Calculate Mean Squared Log Error.
        
        Arguments:
            actuals (pd.DataFrame): a dataframe containing the actual values, with columns 'month_id', 'country_id', 
            and the target variable (e.g. 'ucdp_ged_sb_best_sum'), corresponding to self.prediction_target.
        """

        # merge predictions with actuals on month_id and country_id
        merged = self.predictions.merge(
            actuals,
            left_on=["target_month_id", "country_id"],
            right_on=["month_id", "country_id"],
            how="inner"
        )

        # define y_true and y_pred
        y_true = merged[self.prediction_target].to_numpy(dtype=float)
        y_pred = merged[self.prediction_col].to_numpy(dtype=float)

        # check for negative values, as MSLE cannot be computed with negative values
        if np.any(y_true < 0) or np.any(y_pred < 0):
            raise ValueError("MSLE cannot be computed with negative values.")

        return float(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))
    

    def mse(
            self, 
            actuals: pd.DataFrame, 
        ) -> float:
        """
        Calculate Mean Squared Error.
        
        Arguments:
            actuals (pd.DataFrame): a dataframe containing the actual values, with columns 'month_id', 'country_id', 
            and the target variable (e.g. 'ucdp_ged_sb_best_sum'), corresponding to self.prediction_target.
        """

        # merge predictions with actuals on month_id and country_id
        merged = self.predictions.merge(
            actuals,
            left_on=["target_month_id", "country_id"],
            right_on=["month_id", "country_id"],
            how="inner"
        )

        # define y_true and y_pred
        y_true = merged[self.prediction_target].to_numpy(dtype=float)
        y_pred = merged[self.prediction_col].to_numpy(dtype=float)

        # check for negative values, as MSE cannot be computed with negative values
        if np.any(y_true < 0) or np.any(y_pred < 0):
            raise ValueError("MSE cannot be computed with negative values.")

        return float(np.mean((y_true - y_pred) ** 2))


    def __repr__(self) -> str:
        return (
            f"Prediction({self.train_split}, "
            f"{self.test_split}, "
            f"Target: {self.prediction_target}, as column {self.prediction_col}, "
            f"Size={self.predictions.shape})"
        )

class DynamicModel:

    """
    Class for a Dynamic model that can be trained on a specific train split
    """

    def __init__(
            self, 
            train_split: TrainSplit,
            model: Optional[AnyModel] = None,
            random_state: int = 42
        ):

        self.train_split = train_split
        self.model = model if model is not None else RandomForestRegressor(random_state=random_state)
        
        self.features = []
        self.target = ""

        self.train_shape_ = (0, 0)
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
        y = data["target"]

        self.train_shape_ = X.shape

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
            full_split: Tuple[int, int],
            random_state: int = 42
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
        self.random_state = random_state

        self._is_fitted = False

    def fit(self) -> None:

        train_splits = self.get_train_splits()

        start_time = time.time()
    
        for split in tqdm(train_splits, total = len(train_splits), desc="Fitting models"):
            
            # init model
            model = DynamicModel(
                train_split=split, random_state=self.random_state
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
                    
                    test_data = data.copy()

                    # lead the target variable by the step size
                    test_data["target_month_id"] = test_data["month_id"] + test_split.step

                    # filter the data to only include rows where the month_id is within the test split
                    test_data = test_data[test_data['target_month_id'].between(test_split.start_month, test_split.end_month)]

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
                        predictions=predictions_df,
                        prediction_target=self.target
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
            end_month = month_min + self.train_window_size - 1

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
        start_month = model.train_split.start_month - test_window_size
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

