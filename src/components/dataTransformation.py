from dataclasses import dataclass
import os
import sys
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator,TransformerMixin
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

def handle_item_identifer(item_identifers):
    try:
        logging.info("Handling item identifier")
        col_1 = []
        col_2 = []
        col_3 = []
        col_1_map = {'DR':0, 'FD':1, 'NC':2}
        for i in range(len(item_identifers)):
            col_1.append(col_1_map[item_identifers[i][:2]])
            col_2.append(ord(item_identifers[i][2:3])-64)
            col_3.append(int(item_identifers[i][3:]))
        logging.info('Leaving handle item identifier')
        return (col_1,col_2,col_3)
    except Exception as e:
        CustomException(e,sys)

def handle_item_fat_Content(item_fat_content: list[str] ) -> list[int]:
    try:
        logging.info("Handling item fat")
        mapped = []
        for i in range(len(item_fat_content)):
            if item_fat_content[i][0]=='L' or item_fat_content[i][0]=='l':
                mapped.append(0)
            else:
                mapped.append(1)
        return mapped
    except Exception as e:
        raise CustomException(e,sys)

@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join('artifacts','preprocessor.pkl')


class MiniProcessor(BaseEstimator,TransformerMixin):
    
    def __init__(self):
        pass   

    def fit_transform(self, X,y=None):
        try:
            logging.info('entered mini preprocessor Handling the item identifier and fat content data')
            col_1,col_2,col_3 = handle_item_identifer(X.loc[:,'Item_Identifier'])
            fat_content = handle_item_fat_Content(X.loc[:,'Item_Fat_Content'])
            X['Item_Identifier_1'] = col_1
            X['Item_Identifier_2']  = col_2
            X['Item_Identifier_3'] = col_3
            X['Item_new_Fat_Content'] =  fat_content
            X.drop(['Item_Identifier','Item_Fat_Content'],axis=1)
            logging.info('Completed mini preprocessor.')
            return X
        except Exception as e:
            CustomException(e,sys)
    
    def transform(self, X):
        
        try:
            logging.info('Handling the item identifier and fat content data')
            col_1,col_2,col_3 = handle_item_identifer(X['Item_Identifier'])
            fat_content = handle_item_fat_Content(X['Item_Fat_Content'])
            X['Item_Identifier_1'] = col_1
            X['Item_Identifier_2']  = col_2
            X['Item_Identifier_3'] = col_3
            X['Item_new_Fat_Content'] =  fat_content
            X.drop(['Item_Identifier','Item_Fat_Content'],axis=1,inplace=True)
            return X
        except Exception as e:
            CustomException(e,sys)
    
    


class DataTransformation:
    
    def __init__(self):
        self.DataTransformerConfig = DataTransformationConfig()

    def initateDataTransform(self,train_data_path,test_data_path):
        try:
            logging.info('Dividing the data into input and target')
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)
            input_train = train_data.drop(['Item_Outlet_Sales'],axis=1)
            train_target = train_data['Item_Outlet_Sales']
            input_test = test_data
            logging.info('Performing the transformation on the data')
            preprocessor = self.getDataTransformer()
            transfromed_train = preprocessor.fit_transform(input_train)
            transformed_train = np.c_[transfromed_train,train_target]
            transformed_test = preprocessor.transform(input_test)
            save_object(file_path= self.DataTransformerConfig.preprocessor_path,object=preprocessor)
            return (transformed_train,transformed_test,self.DataTransformerConfig.preprocessor_path) 
        except Exception as e:
            raise CustomException(e,sys)
            



    def getDataTransformer(self):
        
        try:
            
            new_numerical_cols = ['Item_Weight','Item_Identifier_1','Item_Identifier_2','Item_Identifier_3',
                                'Item_new_Fat_Content','Item_Visibility','Outlet_Establishment_Year','Item_MRP',]
            
            new_categorical_cols = ['Item_Type', 'Outlet_Identifier',
                                'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
            
            
            logging.info('Building the numerical data transformer')
            numerical_transformer = Pipeline(
                    steps = [
                        ('numerical_simple_inputer',SimpleImputer(strategy = 'mean')),
                         ('numerical_standard_scalar',StandardScaler())
                    ]
                )
            
            logging.info('Building the categorical data transformer')
            categorical_transformer = Pipeline(
                [
                    ('categorical_simple_inputer',SimpleImputer(strategy='most_frequent')),
                    ('categorical_ordinal_encoder',OrdinalEncoder()),
                    ('categorical_standard_scalar',StandardScaler())
                ]
            )

            logging.info('Building the final preprocessor')
            coltransformer = ColumnTransformer(
                [
                    ('numerical_preprocessor',numerical_transformer,new_numerical_cols),
                    ('categorical_preprocessor',categorical_transformer,new_categorical_cols)
                ]
            )
            logging.info('Final Pipeline')
            preprocessor = Pipeline(
                steps = [
                    ('minipreprocessor',MiniProcessor()),
                    ('Coltransformer',coltransformer)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
        
    
    