import pickle
import dill
import os
import sys
from src.exception import CustomException
from src.logger import logging


def save_object(file_path,object):
    try:
        logging.info(f'Saving the model at f {file_path}')
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name,exist_ok=True)
        with open(file_path ,"wb") as file_obj:
            pickle.dump(object,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    

