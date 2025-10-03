from pymongo import MongoClient
import os
import pika
import json
from customers.nlp import RADARFinancialSummarizer
from general.translator import from_russian_to_english, from_english_to_russian

client = MongoClient("mongodb+srv://andreydem42_db_user:zrCpoM9uRYBH2jQM@finamhackathon.rs3houu.mongodb.net/")
db = client["FinamHackathon"]
collection = db["news"]

url = os.getenv('RABBIT_URL')
# Название очереди, откуда будем читать данные
QUEUE_NAME = "news"

params = pika.URLParameters(url)
connection = pika.BlockingConnection(params)
channel = connection.channel()
summarize = RADARFinancialSummarizer()

channel.queue_declare(queue=QUEUE_NAME, durable=True)


# Здесь бизнес-логика клиента. Эта функция будет вызываться всякий раз, когда RabbitMQ готов получить сообщение.
def get_message(data: dict):
    text = data['text']
    processed_text = summarize.preprocess_text(text)
    english_text = from_russian_to_english(processed_text)
    summary_result = summarize.summarize_news(english_text)
    russian_summary = from_english_to_russian(summary_result['summary'])
    collection.insert_one({
        "summary": russian_summary,
        "link": data['link'],
        "datetime": data['datetime'],
        "title": data['title'],
        "tags": data['tags'],
        "tickers": data['tickers']
    })


def callback(ch, method, properties, body):
    get_message(json.loads(body.decode()))
    ch.basic_ack(delivery_tag=method.delivery_tag)


channel.basic_consume(
    queue=QUEUE_NAME,
    on_message_callback=callback,
    auto_ack=False
)

channel.start_consuming()