import os
import sys

from catboost import CatBoostRegressor
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

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evalute_models

from src.config.config import ModelTrainingCOnfig

class ModelTrainer:
    def __init__(self):
        self.model_training_config= ModelTrainingCOnfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Spliting training and test input data")
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            model = {
                "Random Forest" : RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "k-Neighbors Regression": KNeighborsRegressor(),
                "XGBRegression": XGBRegressor(),
                "CatBoosting Regression": CatBoostRegressor(),
                "AdaBoost Regression": AdaBoostRegressor()
            }

            model_report:dict = evalute_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,
                                               models=model)

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get the model names from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = model[best_model_name]

            if best_model_score<0.6:
                raise CustomException("NO Best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_training_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)

            r2_square = r2_score(y_test,predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)