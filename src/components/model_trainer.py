import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import model_evaluate

@dataclass

class ModelTrainerConfig:
    trained_model_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_training(self,train_array,test_array):

        try:
            logging.info('Split the data into train and test')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:,-1],
                test_array[:, :-1],
                test_array[:,-1],
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            model_report :dict = model_evaluate(X_train, y_train, X_test, y_test, models)
            # Get best score
            best_model_score = max(sorted(model_report.values()))
            # Get best model name
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No good model found!")
            logging.info('Best model on both training and test set')

            save_object(file_path = self.model_trainer_config.trained_model_path, obj = best_model)
            logging.info('Model saved successfully')

            predicted_best_model = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted_best_model)

            return r2


        except Exception as e:
            raise CustomException(e, sys)
        
    
