import os
import sys
import dill
import yaml
from box import ConfigBox
from box.exceptions import BoxValueError
from pathlib import Path
from pandas import DataFrame
import numpy as np
import pandas as pd

from us_visa.exception import CustomException
from us_visa.logger import logging


def read_yaml_file(path_to_yaml: Path) -> ConfigBox:
    logging.info("Entered the read_yaml_file method of utils")
    try:
        with open(path_to_yaml, 'rb') as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"yaml file: {path_to_yaml} loaded successfully")
            print(content)
            return ConfigBox(content)
        logging.info("Exited the read_yaml_file method of utils")  

    except BoxValueError:
        raise ValueError("Yaml file is empty")
        
    except Exception as e:
        raise CustomException(e, sys) from e


def write_yaml_file(path_to_yaml: Path, content: object, replace:bool = False) -> None:
    logging.info("Entered the write_yaml_file method of utils")
    try:
        if replace:
            if os.path.exists(path_to_yaml):
                os.remove(path_to_yaml)
            os.makedirs(os.path.dirname(path_to_yaml), exist_ok=True)
            with open(path_to_yaml, 'w') as yaml_file:
                yaml.dump(content, yaml_file)
            logging.info(f"yaml file: {path_to_yaml} written successfully")
        logging.info("Exited the write_yaml_file method of utils")  

    except Exception as e:
        raise CustomException(e, sys) from e
    

def load_object(path_to_obj: Path) -> object:
    logging.info("Entered the load_object method of utils")
    try:
        with open(path_to_obj, "rb") as file_obj:
            obj = dill.load(file_obj)
            logging.info("Exited the load_object method of utils")
            return obj

    except Exception as e:
        raise CustomException(e, sys) from e


def save_numpy_array_data(path_to_npdata: Path, array: np.array):
    logging.info("Entered the save_numpy_array_data method of utils")
    """
    Save numpy array data to file
    file_to_npdata: str path of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(path_to_npdata)
        os.makedirs(dir_path, exist_ok=True)
        with open(path_to_npdata, 'wb') as file_obj:
            np.save(file_obj, array)
        logging.info("Exited the save_numpy_array_data method of utils")

    except Exception as e:
        raise CustomException(e, sys) from e
    

def load_numpy_array_data(path_to_npdata: Path) -> np.array:
    logging.info("Entered the load_numpy_array_data method of utils")

    """
    load numpy array data from file
    path_to_npdata: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(path_to_npdata, 'rb') as file_obj:
            np_array_data = np.load(file_obj)
            logging.info("Exited the load_numpy_array_data method of utils")
            return np_array_data
        
    except Exception as e:
        raise CustomException(e, sys) from e




def save_object(path_to_obj: Path, obj: object) -> None:
    logging.info("Entered the save_object method of utils")

    try:
        os.makedirs(os.path.dirname(path_to_obj), exist_ok=True)
        with open(path_to_obj, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info("Exited the save_object method of utils")

    except Exception as e:
        raise CustomException(e, sys) from e



def drop_columns(df: DataFrame, cols: list)-> DataFrame:
    logging.info("Entered the drop_columns method of utils")
    """
    drop the columns form a pandas DataFrame
    df: pandas DataFrame
    cols: list of columns to be dropped
    """
    logging.info("Entered drop_columns methon of utils")
    try:
        dataframe = df.drop(columns=cols, axis=1)
        logging.info("Exited the drop_columns method of utils")
        return dataframe
    
    except Exception as e:
        raise CustomException(e, sys) from e



def separate_numerical_and_categorical(df: pd.DataFrame):
    logging.info("Entered the separate_numerical_and_categorical method of utils")
    """
    Separates numerical and categorical features from the given DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    tuple: A tuple containing:
        - numerical_df (pd.DataFrame): DataFrame containing only numerical columns.
        - categorical_df (pd.DataFrame): DataFrame containing only categorical columns.
    """
    try:
        # Select numerical columns
        numerical_df = df.select_dtypes(include=['number'])
        
        # Select categorical columns
        categorical_df = df.select_dtypes(include=['object', 'category'])
        
        return numerical_df, categorical_df
    except Exception as e:
        raise CustomException(e, sys) from e
