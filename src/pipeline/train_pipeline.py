import os
import sys
from src.utils import load_model
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

from sklearn.metrics import r2_score

@dataclass
class ModelTrainConfig():
    model_path = os.path.join('artifacts','model.pkl')
    preprocessor_path  = os.path.join('artifacts','preprocessor.pkl')


class TrainPipeline:

    def __init__(self):
        self.modeltrainconfig = ModelTrainConfig()

    def initate_train_model(self,data):
        
        try:
            logging.info("Loading the models.")
            model = load_model(self.modeltrainconfig.model_path)
            preprocessor = load_model(self.modeltrainconfig.preprocessor_path)
            logging.info('Initate the Preprocessing the data for training.')
            X_train,y_train = data.drop('Item_Outlet_Sales',axis = 1),data['Item_Outlet_Sales']
            X_train_processed = preprocessor.fit_transform(X_train)
            logging.info('Initating the model training on the transformed data.')
            model.fit(X_train_processed,y_train)
            logging.info("Saving the new trained models.")
            save_object(
                file_path = self.modeltrainconfig.model_path,
                object = model
            )
            save_object(
                file_path = self.modeltrainconfig.preprocessor_path,
                object = preprocessor 
            )
            y_pred = model.predict(X_train_processed)
            r2 = r2_score(y_pred= y_pred,y_true=y_train)
            return (preprocessor,model,r2)
        
        except Exception as e:
            raise CustomException(e,sys)
        
