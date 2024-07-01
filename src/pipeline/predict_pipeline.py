import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import load_model

from dataclasses import dataclass

@dataclass
class ModelTestConfig:
    preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
    model_path = os.path.join('artifacts','model.pkl')

class PredictPipeline:

    def __init__(self):
        self.modeltestconfig = ModelTestConfig()

    def initate_predict(self,data):

        try:
            logging.info('Initate the predict pipeline.')
            print(data)
            model = load_model(self.modeltestconfig.model_path)
            preprocessor = load_model(self.modeltestconfig.preprocessor_path)
            logging.info('Initate the data transforming .')
            transformed_data = preprocessor.transform(data)
            logging.info('Initate the model prediction.')
            y_pred = model.predict(transformed_data)
            return y_pred
        except Exception as e:
            raise CustomException(e,sys)
    

    


