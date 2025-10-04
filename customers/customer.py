from pymongo import MongoClient
import os
import pika
import json
from customers.nlp import process_single_text
from datetime import datetime

client = MongoClient(os.getenv("MONGO"))
db = client["FinamHackathon"]
collection = db["news"]

url = os.getenv('RABBIT_URL')
# Название очереди, откуда будем читать данные
QUEUE_NAME = "news"

params = pika.URLParameters(url)
connection = pika.BlockingConnection(params)
channel = connection.channel()

channel.queue_declare(queue=QUEUE_NAME, durable=True)


# Здесь бизнес-логика клиента. Эта функция будет вызываться всякий раз, когда RabbitMQ готов получить сообщение.
def get_message(data: dict): # see line 55 in trading_view_producer.py
    text = data['text']
    result = process_single_text(text)
    result['datetime'] = datetime.strptime(data['datetime'], "%d.%m.%Y %H:%M")
    result['links'] = data['links']
    companies = set(data['companies'])
    companies.update(result['entities']['companies'])
    tickers = set(data['tickers'])
    tickers.update(result['entities']['tickers'])
    result['entities']['companies'] = list(companies)
    result['entities']['tickers'] = list(tickers)
    del result['parsed_content']
    collection.insert_one(result)

    # давай вот такой json
    # {
    #     "topic": string, - ТЕМА
    #     "draft": {
    #       "title": string, - ЗАГОЛОВОК
    #       "lead": string, - ЛИД
    #       "points": [string], - КЛЮЧЕВЫЕ ПУНКТЫ
    #     },
    #     "sentiment": float, - СЕНТИМЕНТ
    #     "hotness": float, - ГОРЯЧЕСТЬ
    #     "why_hot": string, - Причина
    #     "why_now": string, - ПОЧЕМУ СЕЙЧАС
    #     "impact_analysis": string, - АНАЛИЗ ВЛИЯНИЯ
    #     "statistics": {
    #         "compression": float, - СЖАТИЕ
    #         "quality": float, - Качество
    #         "processing_started": datetime, - Обработка начата
    #         "processing_ended": datetime, - Обработка завершена
    #     },
    #     "entities": {
    #         "companies": [string], - дополняем из data
    #         "tickers": [string], - дополняем из data
    #         "countries": [string],
    #         "sectors": [string],
    #         "currencies": [string],
    #     },
    #     "datetime": datetime, - из data (только она приходит строкой, нужно перевести в datetime)
    #     "links": [string], - из data
    # }
    #
    # если текст короткий (<500 символов), тогда не делаем lead, statistics, points


def callback(ch, method, properties, body):
    get_message(json.loads(body.decode()))
    ch.basic_ack(delivery_tag=method.delivery_tag)


channel.basic_consume(
    queue=QUEUE_NAME,
    on_message_callback=callback,
    auto_ack=False
)

channel.start_consuming()