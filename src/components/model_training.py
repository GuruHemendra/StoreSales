import pandas as pd
import numpy as np
import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from sklearn.linear_model import (Lasso,Ridge)
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor)
from xgboost import XGBRegressor

from dataclasses import dataclass

from src.utils import save_object,train_model

@dataclass
class ModelClassConfig:
    best_model_path = os.path.join('artifacts','model.pkl')
    all_model_path = os.path.join('artifacts','all_trained_models')


class ModelTrainer:

    def __init__(self):
        self.modelclassconfig = ModelClassConfig()

    def initate_model_trainer(self,train_data,test_data):
        
        try:
            logging.info('Initiating the model building and training process.')
            models = {
                'Lasso': Lasso(),
                'Ridge' : Ridge(),
                'SVR': SVR(),
                'RandomForest' : RandomForestRegressor(),
                'DecisionTree' : DecisionTreeRegressor(),
                # 'GradientBoost' : GradientBoostingRegressor(),
                'AdaBoost' : AdaBoostRegressor(),
                'XGBoost' : XGBRegressor(),
            }

            grid_search_params = {

                'Lasso': {
                    'alpha': [0.1, 0.2, 0.5, 1, 2],
                    'max_iter': [500, 700, 1000, 1500],
                    'random_state': [42],
                },

                'Ridge': {
                    'alpha': [0.1, 0.2, 0.5, 1, 1.5],
                    'max_iter': [200, 500, 700, 1000, 1500],
                    'random_state': [42],
                },

                'SVR': {
                    'C': [1, 0.5, 2, 1.5],
                    'epsilon': [0.1, 0.01, 0.05, 0.15],
                    'kernel': ['rbf', 'linear'],
                },

                'DecisionTree': {
                    'max_depth': [16, 32, 64, 128],
                    'max_leaf_nodes': [16, 32, 128, 64],
                    'min_samples_split': [5, 10, 20, 50],
                },

                'RandomForest': {
                    'max_depth': [50, 100, 200, 300],
                    'max_leaf_nodes': [16, 32, 64, 128],
                    'min_samples_split': [10, 20, 30],
                },

                'GradientBoost': {
                    'learning_rate': [0.1, 0.01, 0.5, 0.2],
                    'max_depth': [16, 32, 64],
                    'min_samples_leaf': [10, 20, 50, 100],
                    'n_estimators': [50, 100, 150, 200, 300],
                    'random_state': [42],
                },

                'AdaBoost': {
                    'n_estimators': [100, 200, 300, 500, 1000],
                    'learning_rate': [0.1, 0.5, 1, 1.2, 0.7, 0.01],
                    'random_state': [42],
                },

                'XGBoost': {
                    'learning_rate': [0.1, 0.2, 0.5, 1],
                    'n_estimators': [100, 200, 300, 500, 1000],
                    'max_depth': [16, 32, 64],
                    'gamma': [1, 2, 0.1, 0.5],
                },

            }
            logging.info('Spliting the data into train target and input.')
            X_train = train_data[:,:-1]
            X_test = test_data[:,:-1]
            y_train = train_data[:,-1]
            y_test = test_data[:,-1]

            cross_validation = 5
            scoring = 'r2'
            logging.info('Calling for the model to train')
            report  = train_model(models = models, X_train= X_train, X_test= X_test,
                                  y_train=y_train,y_test = y_test,
                                  grid_params= grid_search_params,
                                  scoring= scoring,cv= cross_validation)


            # grid_params= grid_search_params, 
            #                      scoring = scoring, cv=cross_validation
            best_model_name = list(models.keys())[list(report.values()).index(max(list(report.values())))]
            best_model = models[best_model_name]
            logging.info("Saving all trained models")
            for model_name in models.keys():
                logging.info(model_name,report[model_name])
                save_object(
                    file_path= os.path.join(self.modelclassconfig.all_model_path,f'{model_name}.pkl'),
                    object= models[model_name]
                )
            
            logging.info(f'The best model is {best_model_name}')
            logging.info('Saving the best model')
            save_object(
                file_path= self.modelclassconfig.model_path,
                object= best_model
            )
            logging.info(f"Best Model parameters are : {best_model.get_params()}")
            return (best_model_name,best_model,report[best_model_name])
        
        except Exception as e:
            raise CustomException(e,sys)
    
