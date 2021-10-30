from flask import Flask
from flask_restx import Api
from .ml_models import MLModelsDAO
from log import log

application = Flask(__name__)
api = Api(application)

models_dao = MLModelsDAO()

from app import views
