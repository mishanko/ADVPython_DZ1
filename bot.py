import telebot
import requests
from config import API_KEY, API_URL
from helpers import train, retrain, data_for_prediction, predict

bot = telebot.TeleBot(API_KEY)

@bot.message_handler(commands=['start'])
def welcome(message):
    """Приветственное сообщение с инструкцией
    """
       
    start_string = """Привет!

    Это бот для реализации функционала ML API.
    Что он умеет:

    1) /start - Отображение справки.

    2) /models - Отображение, моделей, доступных для обучения и списка их гиперпараметров.

    3) /train - Начало обучение и инструкция для обучения.

    4) /predict - Начало прогнозирования и инструкция для прогнозирования.

    5) /retrain - Начало переобучения и инструкция для переобучения.

    """
    
    bot.send_message(message.chat.id, start_string)

# * Просмотр моделей, доступных для обучения
@bot.message_handler(commands=['models'])
def available_models(message):  
    """Сообщение с описанием имеющихся моделей и их гиперпараметров
    """
    res = requests.get(f'{API_URL}/api/ml_models').json()
    LR_id = res[0]['id']
    LR_h = res[0]['hyperparameters']
    DT_id = res[1]['id']
    DT_h = res[1]['hyperparameters']
    models_string = f"""Доступные для обучения модели:

    ID: {LR_id}
    CLASS: Logistic Regression🦭
    HYPERPARAMETERS:{list(LR_h.keys())}

    ID: {DT_id}
    CLASS: Decision Tree🌲
    HYPERPARAMETERS:{list(DT_h.keys())}
    """
    bot.send_message(message.chat.id, models_string)    

# * Обучение модели
@bot.message_handler(func=lambda message: message.text=='/train', commands=['train'])
def model_choose(message):  
    """Сообщение с инструкцией по обучению
    """
    
    res = requests.get(f'{API_URL}/api/ml_models').json()
    LR_id = res[0]['id']
    DT_id = res[1]['id']
    train_string = f"""Модели, доступные для обучения:

    ID: {LR_id}
    CLASS: Logistic Regression🦭
    
    ID: {DT_id}
    CLASS: Decision Tree🌲

    *Формат данных:
        col1|col2|...|target
        Данные в формате csv
    """
    bot.send_message(message.chat.id, train_string)    
    bot.send_message(message.chat.id, "Загрузите данные") 

@bot.message_handler(content_types=['document'])
def file_reciever(message):
    """Сообщение на получение файла
    """
    
    file = bot.get_file(message.document.file_id)
    down = bot.download_file(file.file_path)
    with open('data/train.csv', 'wb') as new_file:
        new_file.write(down)   
    bot.send_message(message.chat.id, 'Document received, sir!')
    file_string = """Если вы хотите обучить модель, введите через пробел:

    train {ID модели} {hypers}
    *hypers(optional) - словарь с гиперпараметрами. (Пример: {'max_iter':1000})

    Если вы хотите переобучить модель, введите через пробел:

    retrain {ID модели} {Номер модели} {hypers}
    *hypers(optional) - словарь с гиперпараметрами. (Пример: {'max_iter':1000})
    """
    bot.send_message(message.chat.id, f'{file_string}')

@bot.message_handler(func=lambda message: message.text.split()[0]=='train', content_types=['text'])
def training(message):
    """Сообщение на обучение модели"""
    bot.send_message(message.chat.id, "Процесс обучения...")    
    if len(message.text.split()) > 2:# проверка на наличиние гиперпараметров
        id_ = int(message.text.split()[1])
        hypers = message.text.split()[2].replace("'", '"')
        train(id_, 'data/train.csv', hypers)
    else:
        id_ = int(message.text.split()[1])
        train(id_, 'data/train.csv')
    res = requests.get(f'{API_URL}/api/ml_models/{id_}').content
    bot.send_message(message.chat.id, f"Обученные модели:\n{eval(res)}")    


# * Предсказание модели
@bot.message_handler(func=lambda message: message.text=='/predict', commands=['predict'])
def choose_model(message):  
    """Сообщение на инструкцию по предсказанию"""
    res = requests.get(f'{API_URL}/api/ml_models/1').content
    predict_string = f"""Модели, доступные для предсказания:
    {eval(res)}

    Для получения предсказания введите:

    predict (ID модели) (Номер модели) (data)

    *data - вектор данных через пробел (Пример: 5.1 3.5 1.4 0.2)
    """
    bot.send_message(message.chat.id, predict_string) 

@bot.message_handler(func=lambda message: message.text.split()[0]=='predict', content_types=['text'])
def prediction(message):
    """Сообщение на отображение предсказания"""
    msg = message.text.split()
    id_ = int(msg[1])
    num = str(msg[2])
    data = msg[3:]
    data = data_for_prediction(data)
    bot.send_message(message.chat.id, 'Анализируем…')  
    preds = predict(message, id_, num, data)
    bot.send_message(message.chat.id, f'Предсказание: {eval(preds)["Prediction"]}')    


# * Переобучение модели
@bot.message_handler(func=lambda message: message.text=='/retrain', commands=['retrain'])
def choose_model(message):  
    """Сообщение на инструкцию переобучения"""
    
    res = requests.get(f'{API_URL}/api/ml_models/1').content
    retrain_string = f"""Модели, доступные для переобучения:

    {eval(res)}

    *Формат данных:
        col1|col2|...|target
        Данные в формате csv
    """
    bot.send_message(message.chat.id, retrain_string)  
    bot.send_message(message.chat.id, "Загрузите данные")  

@bot.message_handler(func=lambda message: message.text.split()[0]=='retrain', content_types=['text'])
def retraining(message):
    """Сообщение на отображение переобучения"""

    bot.send_message(message.chat.id, "Процесс переобучения...") 
    if len(message.text.split()) > 2: # проверка на наличиние гиперпараметров
        id_ = int(message.text.split()[1])
        num = int(message.text.split()[2])
        hypers = message.text.split()[3].replace("'", '"')
        retrain(id_, num, 'data/train.csv', hypers)
    else:
        id_ = int(message.text.split()[1])
        num = int(message.text.split()[2])
        retrain(id_, num, 'data/train.csv')
    res = requests.get(f'{API_URL}/api/ml_models/{id_}').content
    bot.send_message(message.chat.id, f"Trained models: {eval(res)}")  

bot.polling(non_stop=True, skip_pending=True)    
