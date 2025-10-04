import math
import telebot
import os
import re
from parse import *
from datetime import datetime
from pymongo import MongoClient

from general.settings import TIMEZONE

TOKEN = os.getenv('BOT_TOKEN')
bot = telebot.TeleBot(TOKEN, parse_mode='MARKDOWNV2')

client = MongoClient(os.getenv('MONGO'))
db = client["FinamHackathon"]
collection = db["news"]

NEWLINE = '\n'

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    reply = """
🔥 Привет! Я помогу тебе узнать самые горячие новости финансового мира. 🔥
Больше никаких скучных и малозначащих текстов — только самые топовые события, двигающие рынок! 🚀

Возможные команды:
`03.10.2025-05.10.2025 3` — получить топ-3 новости за период
`today 3` — получить топ-3 новости за сегодня
`/filter 03.10.2025-05.10.2025 5 apple, aapl, США` — фильтр по ключевым словам за период
`/filter today 5 apple aapl США` — фильтр по ключевым словам за сегодня
    """
    bot.send_message(message.chat.id, escape_markdown_v2(reply).replace('\`', '`'))


@bot.message_handler(commands=['filter'])
def filter_news(message):
    try:
        # удаляем префикс команды
        rest = message.text[len('/filter'):].strip()
        if not rest:
            raise TypeError()
        # today-режим
        if rest.startswith('today'):
            # форматы: "today k keywords..." или "today k keyword1, keyword2"
            tokens = rest.split(maxsplit=2)
            if len(tokens) < 2:
                raise TypeError()
            k = int(tokens[1])
            keywords_part = tokens[2] if len(tokens) > 2 else ''
            start_date = datetime.now(tz=TIMEZONE).replace(hour=0, minute=0, second=0)
            end_date = datetime.now(tz=TIMEZONE).replace(hour=23, minute=59, second=59)
        else:
            # ожидается: "dd.mm.yyyy-dd.mm.yyyy k keywords..."
            first_space = rest.find(' ')
            if first_space == -1:
                raise TypeError()
            dates = rest[:first_space]
            remainder = rest[first_space+1:].strip()
            parts = remainder.split(maxsplit=1)
            if not parts:
                raise TypeError()
            k = int(parts[0])
            keywords_part = parts[1] if len(parts) > 1 else ''
            d1, d2 = map(str.strip, dates.split('-', 1))
            start_date = datetime.strptime(d1, "%d.%m.%Y")
            end_date = datetime.strptime(d2, "%d.%m.%Y").replace(hour=23, minute=59, second=59)
        # нормализация ключевых слов
        raw_keywords = re.split(r"[,\s]+", keywords_part)
        keywords = [kw.strip() for kw in raw_keywords if kw.strip()]
        # убираем ведущий # у каждого ключевого слова и удаляем дубликаты, сохраняя порядок
        keywords = list(dict.fromkeys([kw.lstrip('#') for kw in keywords]))
        if not keywords:
            raise TypeError()
        # составим регэксп по словам (экранируем)
        pattern = "|".join(map(re.escape, keywords))
        query = {
            "datetime": {"$gte": start_date, "$lte": end_date},
            "$or": [
                {"entities.companies": {"$regex": pattern, "$options": "i"}},
                {"entities.tickers": {"$regex": pattern, "$options": "i"}},
                {"entities.countries": {"$regex": pattern, "$options": "i"}},
                {"entities.sectors": {"$regex": pattern, "$options": "i"}},
                {"entities.currencies": {"$regex": pattern, "$options": "i"}},
            ],
        }
        results = list(collection.find(query).sort("hottness.score", -1).limit(k))
        sent = 0
        for result in results:
            bot.send_message(message.chat.id, build_message(result))
            sent += 1
        if sent == 0:
            bot.send_message(message.chat.id, escape_markdown_v2('Ничего не найдено по указанным ключевым словам.'))
    except Exception:
        usage = (
            "Использование команды:\n"
            "/filter today <k> <keywords>\n"
            "/filter <dd.mm.yyyy>-<dd.mm.yyyy> <k> <keywords>\n"
            "Примеры:\n"
            "/filter today 5 apple aapl США\n"
            "/filter 03.10.2025-05.10.2025 5 apple, aapl, США"
        )
        bot.send_message(message.chat.id, escape_markdown_v2(usage))


@bot.message_handler(func=lambda message: True)
def echo_all(message):
    try:
        if message.text.startswith('today'):
            parsed = parse("today {}", message.text)
            start_date = datetime.now(tz=TIMEZONE).replace(hour=0, minute=0, second=0)
            end_date = datetime.now(tz=TIMEZONE).replace(hour=23, minute=59, second=59)
            k = int(parsed[0])
        else:
            parsed = parse("{}-{} {}", message.text)
            start_date = datetime.strptime(parsed[0], "%d.%m.%Y")
            end_date = datetime.strptime(parsed[1], "%d.%m.%Y")
            end_date = end_date.replace(hour=23, minute=59, second=59)
            k = int(parsed[2])
        results = list(collection.find({
            "datetime": {
                "$gte": start_date,
                "$lte": end_date
            }
        }).sort("hottness.score", -1).limit(k))
        for result in results:
            bot.send_message(message.chat.id, build_message(result))
    except TypeError:
        bot.send_message(message.chat.id, escape_markdown_v2('Неверная команда. Список команд можно получить с помощью /help'))


def escape_markdown_v2(text):
    if not isinstance(text, str):
        return text
    # обязательные для экранирования символы MarkdownV2
    symbols = ['\\', '_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for s in symbols:
        text = text.replace(s, '\\' + s)
    return text


def build_message(r):
    fires = math.ceil(r['hottness']['score'] * 10)
    sentiment_emojy = ['😡', '😟', '😐', '😊', '🤩', '🤩'][math.floor((r['sentiment_score'] + 1) * 5 / 2)]
    title = escape_markdown_v2(r['title'].upper())
    datetime_text = escape_markdown_v2(r['datetime'].strftime('%d.%m.%Y %H:%M'))
    lead = escape_markdown_v2(r['lead'])
    key_points = [escape_markdown_v2(point) for point in r['key_points']]
    links = [escape_markdown_v2(link) for link in r['links']]
    impact_analysis = escape_markdown_v2(r['impact_analysis'])
    reasoning = escape_markdown_v2(r['hottness']['reasoning'])
    why_now = escape_markdown_v2(r['why_now'])
    companies = escape_markdown_v2(', '.join(r['entities']['companies'])) if r['entities']['companies'] else 'не указаны'
    r['entities']['tickers'] = list(filter(lambda x: x != 'UNKNOWN', r['entities']['tickers']))
    tickers = escape_markdown_v2(hashtags(r['entities']['tickers'])) if r['entities']['tickers'] else 'не указаны'
    countries = escape_markdown_v2(', '.join(r['entities']['countries'])) if r['entities']['countries'] else 'не указаны'
    sectors = escape_markdown_v2(', '.join(r['entities']['sectors'])) if r['entities']['sectors'] else 'не указаны'
    currencies = escape_markdown_v2(', '.join(r['entities']['currencies'])) if r['entities']['currencies'] else 'не указаны'
    sentiment_score = escape_markdown_v2(str(r['sentiment_score']))
    return f"""
*{title}*
{datetime_text}
""" + \
        (f"""
{lead}
{NEWLINE.join([f'• {point}' for point in key_points])}
""" if r['statistics']['orig_chars'] > 200 else '') +\
f"""
{NEWLINE.join([link for link in links])}

*{impact_analysis}*

❗️ *Горячая новость:* {reasoning}
⏰ *Почему сейчас:* {why_now}

Оценка горячести: """ + escape_markdown_v2('🔥' * fires) + f"""
Сентимент: {sentiment_score} {sentiment_emojy}

Компании: {companies}
Тикеры: {tickers}
Страны: {countries}
Сектора: {sectors}
Валюты: {currencies}
    """


def hashtags(array):
    return ' '.join(list(map(lambda x: '#' + x, array)))

bot.infinity_polling()