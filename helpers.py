import json
import requests
import pandas as pd
import numpy as np
from config import API_URL

headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

def train(id:int, data:pd.DataFrame, hypers:dict=None):
    """Функция для обработки входных данных и отправления PUT запроса на обучение.

    Args:
        id (int): ID модели
        data (pd.DataFrame): Данные для обучения
        hypers (dict, optional): Словарь с гиперпараметрами. Defaults to None.
    """
    data = pd.read_csv(data)
    X = data[data.columns[:-1]].values.tolist() # все колонки, кроме таргет
    y = data['target'].values.tolist()

    # Отработка момента отсутствия гиперпараметров
    if hypers is not None:
        hypers = eval(hypers)
        data = json.dumps({'X':X, 'y':y, 'H':hypers}, indent=4)
    else:
        data = json.dumps({"X":X, "y":y}, indent=4)

    requests.put(f'{API_URL}/api/ml_models/{id}/service', data=data, headers=headers)

def retrain(id:int, num:str, data:pd.DataFrame, hypers:dict=None):
    """Функция для обработки входных данных и отправления PUT запроса на переобучение.

    Args:
        id (int): ID модели
        num (str): Номер обученной модели
        data (pd.DataFrame): Данные для переобучения
        hypers (dict, optional): Гиперпараметры. Defaults to None.
    """
    data = pd.read_csv(data)
    X = data[data.columns[:-1]].values.tolist() # все колонки, кроме таргет
    y = data['target'].values.tolist()
    
    # Отработка момента отсутствия гиперпараметров
    if hypers is not None:
        hypers = eval(hypers)
        data = json.dumps({'X':X, 'y':y, 'H':hypers, 'num':num}, indent=4)
    else:
        data = json.dumps({"X":X, "y":y, 'num':num}, indent=4)

    requests.put(f'{API_URL}/api/ml_models/{id}/retrain', data=data, headers=headers)

def data_for_prediction(data:list)->np.array:
    """Вспомогательная функция для формирование данных для предсказания

    Args:
        data (list): Данные для предсказания

    Returns:
        np.array: Данные в формате numpy массива
    """
    data = np.array(data).astype(np.float64)
    return data

def predict(id:int, num:str, data:np.array)->dict:
    """Функция для обработки входных данных и отправления POST запроса на предсказание

    Args:
        id (int): ID модели
        num (str): Номер обученной модели
        data (np.array): Данные для предсказания

    Returns:
        dict: Словарь с предсказанием
    """
    data = data.tolist()
    data = json.dumps({'X':data, 'num':num}, indent=4)
    preds = requests.post(f'{API_URL}/api/ml_models/{id}/service', data=data, headers=headers).content  
    return preds