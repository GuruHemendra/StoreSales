import pickle
import dill
import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path,object):
    try:
        logging.info(f'Saving the model at f {file_path}')
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name,exist_ok=True)
        with open(file_path ,"wb") as file_obj:
            pickle.dump(object,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    

def train_model(models,X_train,y_train,X_test,y_test,grid_params=None,scoring=None,cv=None):
    try:
        model_names = models.keys()
        result = {}
        for model_name in model_names:
            model = models[model_name]
            logging.info(f'Starting the Grid Search CV for the {model_name}.')
            if grid_params != None :
                grid = GridSearchCV(
                        estimator= model,
                        param_grid= grid_params[model_name],
                        scoring=scoring,
                        cv = cv
                    )
                grid.fit(X=X_train,y=y_train)
                model.set_params(**grid.best_params_)
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_true=y_test,y_pred=y_pred)
            logging.info(f'The traininig of the {model_name} is completed with r2 = {r2}.')
            logging.info(f"{model_name} parameters are : {model.get_params()}")
            result[model_name] = r2
        return result
    except Exception as e:
        raise CustomException(e,sys)


def load_model(file_path):
    
    try:
        logging.info(f'Loading the model from {file_path}')
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e,sys)
    

def CustomDataSetTrain( Item_Identifier,Item_Weight,Item_Fat_Content,Item_Visibility,
                        Item_Type,Item_MRP,Outlet_Identifier,Outlet_Establishment_Year,
                        Outlet_Size,Outlet_Location_Type,Outlet_Type,Item_Outlet_Sales):
    try:
        dataframe = pd.DataFrame(
                                {   
                                    'Item_Identifier': [Item_Identifier],
                                    'Item_Weight' : [Item_Weight],
                                    'Item_Fat_Content' : [Item_Fat_Content],
                                    'Item_Visibility' : [Item_Visibility],
                                    'Item_Type' : [Item_Type],
                                    'Item_MRP' : [Item_MRP],
                                    'Outlet_Identifier' : [Outlet_Identifier],
                                    'Outlet_Establishment_Year' : [Outlet_Establishment_Year],
                                    'Outlet_Size' : [Outlet_Size],
                                    'Outlet_Location_Type' : [Outlet_Location_Type],
                                    'Outlet_Type' : [Outlet_Type],
                                    'Item_Outlet_Sales' : [Item_Outlet_Sales]
                                }
                            )
        return dataframe
    except Exception as e:
        raise CustomException(e,sys)


def CustomDataSetTest(  Item_Identifier,Item_Weight,Item_Fat_Content,Item_Visibility,
                        Item_Type,Item_MRP,Outlet_Identifier,Outlet_Establishment_Year,
                        Outlet_Size,Outlet_Location_Type,Outlet_Type):
    try:
        dataframe = pd.DataFrame(
                                {   
                                    'Item_Identifier': [Item_Identifier],
                                    'Item_Weight' : [Item_Weight],
                                    'Item_Fat_Content' : [Item_Fat_Content],
                                    'Item_Visibility' : [Item_Visibility],
                                    'Item_Type' : [Item_Type],
                                    'Item_MRP' : [Item_MRP],
                                    'Outlet_Identifier' : [Outlet_Identifier],
                                    'Outlet_Establishment_Year' : [Outlet_Establishment_Year],
                                    'Outlet_Size' : [Outlet_Size],
                                    'Outlet_Location_Type' : [Outlet_Location_Type],
                                    'Outlet_Type' : [Outlet_Type],
                                }
                            )
        return dataframe
    except Exception as e:
        raise CustomException(e,sys)

