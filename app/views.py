from flask_restx import Resource
from app import api, models_dao

import logging
from typing import Tuple, Union, NoReturn

from sklearn.linear_model import LogisticRegression as LR
from sklearn.tree import DecisionTreeClassifier as DT
import numpy as np

log = logging.getLogger(__name__)

# * Словарь моделей, доступных для обучения
models = {1:LR,
          2:DT}

@api.route('/api/ml_models')
class MLModels(Resource):
    """Класс для отображения списка доступных для обучения моделей
       и гиперпараметров
    """
    def get(self) -> list:
        log.info(f'INFO: Trained models = {models_dao._trained_models}')
        return models_dao._ml_models


@api.route('/api/ml_models/<int:id>')
class MLModel(Resource): 
    """Класс для отображения конкретной модели
       и её гиперпараметров
    """

    def get(self, id:int) -> Union[dict, NoReturn]:
        try:
            log.info(f'INFO: Trained models = {models_dao._trained_models[id]}')
            return models_dao.get(id)[1]
        except KeyError as e:
            log.error("ERROR: Entered invalid model number")
            api.abort(404, e)


@api.route('/api/ml_models/<int:id>/service')
class MLModelService(Resource): 
    """Класс для обучения и предсказания
    """

    def get(self, id:int) -> Union[Tuple[dict, dict], NoReturn]:
        try:
            name = models_dao._ml_models[id-1]["name"]
            trained = models_dao._ml_models[id-1]["trained"]
            log.info(f'INFO: Trained models = {models_dao._trained_models[id]}')
            log.info(f'INFO:\nModel id = {id}\nclass = {name}\ntrained = {trained}')
            return models_dao.get(id)[0], 200
        except IndexError as e:
            log.error("ERROR: Entered invalid model number")
            api.abort(404, e)

    def post(self, id:int, num:str=None, data:np.array=None) -> Tuple[dict, int]: 
        id_ = int(id)
        prediction = self._predict(id_, num, data)
        log.info(f"INFO: Here is the prediction: {prediction}")
        return prediction, 200

    def put(self, id:int) -> Tuple[dict, int]:
        model = self._train(id)
        self._save_model(model, id)
        data = {'trained': True} 
        return models_dao.update(id, data), 200

    def delete(self, id:int) -> Union[Tuple[str, int], NoReturn]:
        try: 
            df = api.payload
            num = df['num']
            if num in models_dao._trained_models[id].keys():
                models_dao.delete(id, num)
                log.info(f"INFO: model {num} deleted")
                return '', 204
            else:
                api.abort(404, "WARNING: No model with this number")

        except KeyError as e:
            log.error("ERROR: Invalid model number")
            api.abort(404, e)

    def _train(self, id:int) -> Union[DT, LR, NoReturn]:
        log.info("INFO: Preparing to train...",)
        try: 
            df = api.payload
            print(df)
            if 'H' in df.keys():
                hypers = df['H']
                model = models[id](**hypers)
            else:
                model = models[id]()
            X = df['X']
            y = df['y']
            log.info("INFO: Start training",)
            model.fit(X, y)
            log.info("INFO: Training finished!")
            return model
        except KeyError or AttributeError as e:
            log.error("ERROR: Looks like you either forget X or y values")
            api.abort(404, e)

    def _predict(self, id:int, num:int=None, data:np.array=None) -> Union[dict, NoReturn]:
        try:
            if data is not None:
                X_new = data
                model = models_dao._trained_models[id][num]
            else:
                df = api.payload
                number = df['num']
                model = models_dao._trained_models[id][number]
                X_new = np.fromiter(df['X'], dtype=float)
            
            prediction = {'Prediction': str(model.predict([X_new]))}
            return prediction
        except KeyError as e:
            log.error("ERROR: Invalid model number or No trained models")
            api.abort(404, e)

    def _save_model(self, model:Union[DT, LR], id:int):
        log.info(f"INFO: Saving the model...")
        models_dao._trained_models[id][str(models_dao.num)] = model
        log.info(f"INFO: Trained models: {models_dao._trained_models}")
        models_dao.num += 1


@api.route('/api/ml_models/<int:id>/retrain')
class MLModelRetrain(Resource): 
    """Класс для переобучения модели
    """

    def get(self, id:int) -> Tuple[dict, dict]:
        try:
            name = models_dao._ml_models[id-1]["name"]
            trained = models_dao._ml_models[id-1]["trained"]
            log.info(f'INFO:\nModel id = {id}\nclass = {name}\ntrained = {trained}')
            return models_dao.get(id)[0], 200
        except IndexError as e:
            log.error("ERROR: Enter the number of model you want to use for prediction")
            api.abort(404, e)

    def put(self, id:int) -> Tuple[dict, int]:
        model = self._train(id)
        self._save_model(model, id)
        data = {'trained': True} 
        return models_dao.update(id, data), 200

    def _train(self, id:int) -> Union[DT, LR]:
        log.info("INFO: Prepating to retrain...",)
        try: 
            df = api.payload
            if 'H' in df.keys():
                hypers = df['H']
                model = models[id](**hypers)
            else:
                model = models[id]()
            X = df['X']
            y = df['y']
            self.num = str(df['num'])
            log.info("INFO: Start training",)
            model.fit(X, y)
            log.info("INFO: Training finished!")
            return model
        except KeyError as e:
            log.error("ERROR: Looks like you forget X or num values")
            api.abort(404, e)

    def _save_model(self, model:Union[DT, LR], id:int):
        models_dao._trained_models[id][self.num] = model
        log.info(models_dao._trained_models)