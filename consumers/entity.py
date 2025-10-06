import os
from datetime import datetime
from pymongo import MongoClient

client = MongoClient(os.getenv("MONGO"))
db = client["FinamHackathon"]
collection = db["news"]

data = {
    "topic": "Прорывные технологии",
    "draft": {
      "title": "Генерирация котиков с помощью ИИ",
      "lead": "Apple создала технологию для мгновенной генерации котиков",
      "points": [
        "Генерация проходит за 2 секунды",
        "Один котик будет стоить 0.01$",
        "Технология уже доступна всем желающим"
      ]
    },
    "sentiment": 1.0,
    "hotness": 0.7,
    "why_hot": "ИИ это перспективное направление, а котиками интересуются все",
    "why_now": "Новость вызвала манипулятивные движения на рынке",
    "impact_analysis": "На фоне этой новости и недавних отчетов акции Apple могут вырасти",
    "statistics": {
        "compression": 0.5,
        "quality": 1.0,
        "processing_started": datetime(2024, 6, 1, 12, 48),
        "processing_ended": datetime(2024, 6, 1, 15, 37)
    },
    "entities": {
        "companies": ['Apple'],
        "tickers": ['AAPL', 'WHAT', 'THE', 'FUCK'],
        "countries": ['USA'],
        "sectors": ['IT'],
        "currencies": ['USD'],
    },
    "datetime": datetime(2024, 5, 30, 9, 12),
    "links": ["https://catscaptureworld.com", 'http://meowmeow.com'],
}

collection.insert_one({
    "sentiment_score": -0.5,
    "hottness": {
        "score": 0.2
    },
    "entities": {
        'companies': ['МВФ', 'ЕБРР'],
        'tickers': ['AAPL']
    }
})
# collection.insert_one(data)
