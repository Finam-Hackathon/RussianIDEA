import re
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
import asyncio
from telethon import TelegramClient
from telethon.tl.types import MessageMediaPhoto, MessageMediaDocument
import emoji
import strip_markdown
import pytz
import os
import pika
import json

from general.settings import TIMEZONE, in_period, DAYS_TO_EXPIRE

url = os.getenv('RABBIT_URL')

QUEUE_NAME = "news"

params = pika.URLParameters(url)
connection = pika.BlockingConnection(params)
channel = connection.channel()

channel.queue_declare(queue=QUEUE_NAME, durable=True)


def send_json(data):
    channel.basic_publish(
        exchange="",
        routing_key=QUEUE_NAME,
        body=data,
        properties=pika.BasicProperties(delivery_mode=2)
    )


class EnhancedTelegramParser:
    def __init__(self, api_id, api_hash, session_name='enhanced_parser'):
        self.client = TelegramClient(session_name, api_id, api_hash)

    def replace_tickers_in_text(self, text: str, tickers: list) -> str:
        """Заменяет тикеры в тексте из формата #TICKER на $TICKER"""
        if not text or not tickers:
            return text

        for ticker in tickers:
            text = re.sub(rf'#{ticker}\b', f'${ticker}', text)

        return text


    def remove_hashtags_smart(self, text: str) -> str:
        """Умное удаление хэштегов с сохранением пунктуации"""
        if not text:
            return ""

        cleaned = re.sub(r'\s*#\w+\s*', ' ', text)

        cleaned = re.sub(r'\s+', ' ', cleaned)

        cleaned = re.sub(r'\s\.', '.', cleaned)  # пробел перед точкой
        cleaned = re.sub(r'\s,', ',', cleaned)  # пробел перед запятой
        cleaned = re.sub(r'\s!', '!', cleaned)  # пробел перед восклицательным
        cleaned = re.sub(r'\s\?', '?', cleaned)  # пробел перед вопросом
        cleaned = re.sub(r'\s;', ';', cleaned)  # пробел перед точкой с запятой
        cleaned = re.sub(r'\s:', ':', cleaned)  # пробел перед двоеточием

        # Убираем пунктуацию в начале строки
        cleaned = re.sub(r'^[.,!?;:\s]+', '', cleaned)

        return cleaned.strip()

    def replace_escaped_quotes(self, text: str) -> str:
        """Заменяет экранирующие кавычки на обычные одинарные"""
        if not text:
            return ""

        cleaned = text.replace('\"', "'")

        return cleaned

    def remove_urls(self, text: str) -> str:
        """Удаляет все URL из текста"""
        if not text:
            return ""

        # Удаляем URL-адреса
        cleaned = re.sub(r'https?://\S+', '', text)

        # Убираем лишние пробелы
        cleaned = re.sub(r'\s+', ' ', cleaned)

        return cleaned.strip()

    def _parse_message(self, message) -> Dict[str, Any]:
        """Улучшенный парсинг сообщения с обработкой текста"""

        formatted_date = self._format_date(message.date)

        raw_text = message.text or ""
        processed_text, tags, links = self._process_text(raw_text)

        tickers = self._extract_tickers(raw_text)

        processed_text = self.replace_tickers_in_text(processed_text, tickers)

        processed_text = self.remove_hashtags_smart(processed_text)

        processed_text = self.replace_escaped_quotes(processed_text)

        processed_text = self.remove_urls(processed_text)

        return {
            'id': message.id,
            'date': formatted_date,
            'text': processed_text,
            'views': getattr(message, 'views', 0),
            'forwards': getattr(message, 'forwards', 0),
            'replies': getattr(message, 'replies', {}).get('replies', 0) if getattr(message, 'replies', None) else 0,
            'media_type': self._get_media_type(message.media),
            'has_media': bool(message.media),
            'tags': tags,
            'tickers': tickers,
            'companies': [],  # Можно добавить логику для компаний
            'links': links
        }

    def _format_date(self, date_obj) -> str:
        """Форматирование даты в нужный формат"""
        if not date_obj:
            return ""
        return date_obj.strftime("%d.%m.%Y %H:%M:%S")

    def _process_text(self, text: str) -> tuple:
        """
        Обработка текста: очистка, извлечение тегов и ссылок
        Возвращает: (очищенный_текст, теги, ссылки)
        """
        if not text:
            return "", [], []

        # Сохраняем оригинальный текст для извлечения данных
        original_text = text

        # Извлекаем хэштеги ДО обработки текста
        tags = self._extract_hashtags(original_text)

        # Извлекаем ссылки
        links = self._extract_links(original_text)

        # Очищаем текст
        cleaned_text = self._clean_text(original_text)

        return cleaned_text, tags, links

    def _extract_hashtags(self, text: str) -> List[str]:
        """Извлечение хэштегов"""
        hashtags = re.findall(r'#(\w+)', text)
        return list(set(hashtags))  # Убираем дубликаты

    def _extract_links(self, text: str) -> List[str]:
        """Извлечение ссылок из текста и markdown"""
        # Ссылки в markdown формате [текст](url)
        markdown_links = re.findall(r'\[.*?\]\((https?://[^\s]+)\)', text)

        all_links = list(set(markdown_links))
        return all_links

    def _extract_tickers(self, text: str) -> List[str]:
        """Извлечение тикеров (латинские заглавные буквы, обычно 2-5 символов)"""
        dollar_tickers = re.findall(r'\$([A-Z]{2,5})\b', text)

        hashtag_tickers = re.findall(r'#([A-Z]{2,5})\b', text)

        all_tickers = list(set(dollar_tickers + hashtag_tickers))

        return all_tickers

    def clean_markdown(self, text: str) -> str:
        """Очищает Markdown разметку"""
        return strip_markdown.strip_markdown(text)

    def _clean_text(self, text: str) -> str:
        """Очистка и форматирование текста"""
        if not text:
            return ""

        text = self.clean_markdown(text)

        text = re.sub(r'\n+', '. ', text)
        text = re.sub(r'\s+', ' ', text)

        text = re.sub(r'\.\s*\.', '.', text)

        text = emoji.replace_emoji(text, replace='')

        text = ' '.join(text.split())

        text = text.strip()
        if text and not text.endswith('.'):
            text += '.'

        return text

    def _get_media_type(self, media) -> str:
        """Определение типа медиа"""
        if not media:
            return None
        if isinstance(media, MessageMediaPhoto):
            return 'photo'
        elif isinstance(media, MessageMediaDocument):
            return 'document'
        return 'other'

    async def parse_channel(self, channel_username, limit=1000):
        """Основной метод парсинга"""
        await self.client.start()

        entity = await self.client.get_entity(channel_username)
        messages_data = []

        #print(f"Starting enhanced parsing of {channel_username}...")

        async for message in self.client.iter_messages(entity, limit=limit):
            try:
                message_data = self._parse_message(message)

                crypto_tickers = {'BTC', 'ETH', 'SOL', 'USDT', 'USDC', 'DOGE', 'BNB', 'TON'}
                crypto_tags = {'крипто', 'crypto', 'bitcoin', 'эфириум', 'ethereum'}
                day_tags = {'понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье'}

                if (len(message_data.get('tags', [])) > 0 or
                        len(message_data.get('tickers', [])) > 0):
                    if (not any(tag in crypto_tags for tag in message_data.get('tags', [])) and
                            not any(tag in day_tags for tag in message_data.get('tags', [])) and
                            not any(ticker in crypto_tickers for ticker in message_data.get('tickers', []))):
                        if message_data['text'] != '' and "Bogdanoff Market Research" not in message_data['text']:
                            messages_data.append(message_data)


            except Exception as e:
                print(f"Error processing message {message.id}: {e}")
                continue

        return messages_data

    async def parse_recent_messages(self, channel_username, days=DAYS_TO_EXPIRE, limit=1000):
        await self.client.start()

        since_date = datetime.now() - timedelta(days=days)
        print(f"Парсим сообщения с {since_date.strftime('%d.%m.%Y')}")

        entity = await self.client.get_entity(channel_username)
        messages_data = []

        try:
            async for message in self.client.iter_messages(
                    entity,
                    limit=limit,
                    offset_date=since_date,
                    reverse=True
            ):
                message_data = self._parse_message(message)
                crypto_tickers = {'BTC', 'ETH', 'SOL', 'USDT', 'USDC', 'DOGE', 'BNB', 'TON'}
                crypto_tags = {'крипто', 'crypto', 'bitcoin', 'эфириум', 'ethereum'}
                day_tags = {'понедельник', 'вторник', 'среда', 'четверг', 'пятница', 'суббота', 'воскресенье'}

                if (len(message_data.get('tags', [])) > 0 or
                        len(message_data.get('tickers', [])) > 0):
                    if (not any(tag in crypto_tags for tag in message_data.get('tags', [])) and
                            not any(tag in day_tags for tag in message_data.get('tags', [])) and
                                not any(ticker in crypto_tickers for ticker in message_data.get('tickers', []))):
                        if message_data['text'] != '' and "Bogdanoff Market Research" not in message_data['text']:
                            messages_data.append(message_data)

                if len(messages_data) % 100 == 0:
                    print(f"Получено {len(messages_data)} сообщений")

        except Exception as e:
            print(f"Ошибка при парсинге: {e}")

        await self.client.disconnect()
        print(f"Всего получено {len(messages_data)} сообщений за последние {days} дней")
        return messages_data


class TextProcessor:
    @staticmethod
    def normalize_text(text: str) -> str:
        """Дополнительная нормализация текста"""
        # Заменяем множественные точки
        text = re.sub(r'\.{2,}', '.', text)
        # Убираем пробелы вокруг точек
        text = re.sub(r'\s*\.\s*', '. ', text)
        return text.strip()

    @staticmethod
    def extract_entities(text: str) -> Dict[str, List[str]]:
        """Извлечение различных сущностей из текста"""
        return {
            'hashtags': re.findall(r'#(\w+)', text),
            'mentions': re.findall(r'@(\w+)', text),
            'urls': re.findall(r'https?://[^\s]+', text),
            'emails': re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        }


def parse_and_convert_date(date_str):
    """Парсит строку в формате 'dd.mm.YYYY HH:MM:SS' и конвертирует в московское время"""
    if not date_str:
        return ""

    try:
        utc_dt = datetime.strptime(date_str, "%d.%m.%Y %H:%M:%S")

        utc_dt = pytz.UTC.localize(utc_dt)

        moscow_zone = pytz.timezone('Europe/Moscow')
        moscow_dt = utc_dt.astimezone(moscow_zone)

        return moscow_dt.strftime("%d.%m.%Y %H:%M:%S")

    except Exception as e:
        print(f"Ошибка парсинга даты '{date_str}': {e}")
        return date_str


# Пример
# date_str = "03.10.2024 10:30:00"  # Ваш формат (предположительно UTC)
# moscow_time = parse_and_convert_date(date_str)
# print(moscow_time)  # "03.10.2024 13:30:00" (UTC+3)


async def main():
    api_id = os.getenv('api_id')
    api_hash = os.getenv('api_hash')
    parser = EnhancedTelegramParser(api_id, api_hash)

    messages = await parser.parse_recent_messages('@markettwits', limit=10000)

    json_news_2_weeks = []

    for i in range(len(messages) - 1):
        json_news = {
            'producer': 'markettwits',
            'title': '',
            'text': messages[i]['text'],
            'tags': messages[i]['tags'],
            'tickers': messages[i]['tickers'],
            'companies': [],
            'links': messages[i]['links'],
            'datetime': parse_and_convert_date(messages[i]['date'])[:-3]
        }
        print(json_news)
        json_news_2_weeks.append(json_news)
        send_json(json.dumps(json_news))

    connection.close()



if __name__ == "__main__":
    asyncio.run(main())