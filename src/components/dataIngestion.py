import pandas as pd
import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.components.dataTransformation import DataTransformation
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_path = os.path.join('artifacts','train.csv')
    test_path = os.path.join('artifacts','test.csv')

class DataIngestion:

    def __init__(self):
        self.dataingestionconfig = DataIngestionConfig()

    def initate_data_ingestion(self,train_path,test_path):
        
        try: 
            
            logging.info('Intiating the data ingestion')
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            logging.info('Making the root directory')
            os.makedirs(os.path.dirname(self.dataingestionconfig.train_path),exist_ok=True)
            train_data.to_csv(self.dataingestionconfig.train_path,index=False,header=True)
            test_data.to_csv(self.dataingestionconfig.test_path,index=False,header=True)
            logging.info('Train and Test Data is saved on the artifacts folder.')
            return self.dataingestionconfig.train_path,self.dataingestionconfig.test_path
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':
    ingestion = DataIngestion()
    train_path,test_path = ingestion.initate_data_ingestion('data/Train.csv','data/Test.csv')
    transform = DataTransformation()
    train_data,test_data,_ = transform.initateDataTransform(train_path,test_path)
    print(train_data.shape,test_data.shape)
