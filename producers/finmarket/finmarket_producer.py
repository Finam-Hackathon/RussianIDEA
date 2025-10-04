import time
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import os
import pika
import json

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


def get_page_with_correct_encoding(url):
    response = requests.get(url)

    encodings_to_try = ['utf-8', 'windows-1251', 'cp1251', 'iso-8859-1', 'koi8-r']

    for encoding in encodings_to_try:
        try:
            response.encoding = encoding
            text = response.text
            if 'новости' in text.lower() or 'финмаркет' in text.lower():
                return text
        except:
            continue

    return response.text


def parse_finmarket_news():
    url = 'https://www.finmarket.ru/'
    text = get_page_with_correct_encoding(url)

    if not text:
        print("Не удалось загрузить страницу")
        return []

    soup = BeautifulSoup(text, 'html.parser')
    news_list = []

    news_container = soup.find('div', style=lambda value: value and 'width: 570px' in value)

    if not news_container:
        news_items = soup.find_all('div', style=lambda value: value and 'font-size: 11px' in value)
    else:
        news_items = news_container.find_all('div', style=lambda value: value and 'font-size: 11px' in value)

    date_items = [item for item in news_items if 'года' in item.get_text() and ':' in item.get_text()]

    for date_item in date_items:
        date = date_item.get_text(strip=True)

        title_container = date_item.find_next_sibling('div')
        if not title_container:
            continue

        title_elem = title_container.find('a')
        if not title_elem:
            continue

        title = title_elem.get_text(strip=True)
        link = title_elem.get('href', '')

        annotation_container = title_container.find_next_sibling('div')
        annotation = annotation_container.get_text(strip=True) if annotation_container else ""

        full_link = f"https://www.finmarket.ru{link}" if link.startswith('/') else link

        news_list.append({
            'date': date,
            'title': title,
            'link': full_link,
            'annotation': annotation
        })

    return news_list


def find_specific_news(news_list, keyword):
    for news in news_list:
        if keyword.lower() in news['title'].lower():
            return news
    return None


def parse_companies_reliable(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    companies = []

    target_element = soup.find(string=lambda s: "Компании упоминаемые в новости" in str(s))

    if target_element:
        parent_div = target_element.find_parent('div')
        if parent_div:
            company_links = parent_div.find_all('a')
            for link in company_links:
                companies.append(link.get_text().strip())

    return companies


def clean_news_text(text):
    if not text or not isinstance(text, str):
        return text

    start_patterns = [
        r'^\d{1,2}\s+\w+\.?\s*FINMARKET\.RU\s*-?\s*',  # "3 октября. FINMARKET.RU -"
        r'^\d{1,2}\s+\w+\s+FINMARKET\.RU\s*-?\s*',  # "3 октября FINMARKET.RU -"
        r'^\d{1,2}\.\d{1,2}\.\d{4}\s+FINMARKET\.RU\s*-?\s*',  # "03.10.2024 FINMARKET.RU -"
    ]

    for pattern in start_patterns:
        text = re.sub(pattern, '', text)

    end_patterns = [
        r'\s*Это автоматическое сообщение\.?\s*$',
        r'\s*\(автоматическое сообщение\)\s*$',
        r'\s*Автоматическое сообщение\.?\s*$',
    ]

    for pattern in end_patterns:
        text = re.sub(pattern, '', text)

    return text.strip()


def simple_date_convert(date_string):
    months = {'января': '01', 'февраля': '02', 'марта': '03', 'апреля': '04',
              'мая': '05', 'июня': '06', 'июля': '07', 'августа': '08',
              'сентября': '09', 'октября': '10', 'ноября': '11', 'декабря': '12'}

    parts = date_string.split()
    day = parts[0].zfill(2)
    month = months[parts[1]]
    year = parts[2]
    time = parts[4]

    return f"{day}.{month}.{year} {time}"


if __name__ == "__main__":
    old_json_news = {
            'producer': 'finmarket',
            'title': '',
            'text': '',
            'tags': [],
            'tickers': [],
            'companies': [],
            'link': '',
            'datetime': ''
        }
    while True:
        news = parse_finmarket_news()

        #print(news[0]['link'])
        text = get_page_with_correct_encoding(news[0]['link'])
         #print(text)

        news_text = ""

        if not text:
            print("Не удалось загрузить страницу")
        else:
            soup = BeautifulSoup(text, 'html.parser')
            news_div = soup.find('div', class_='body', itemprop='articleBody')

            if news_div:
                news_text = news_div.get_text(separator=' ', strip=True)
                # news_text = news_div.get_text(separator=' ')
                news_text = clean_news_text(news_text)
                #print(news_text)
            else:
                print("Новость не найдена")

        companies = parse_companies_reliable(text)

        json_news = {
            'producer': 'finmarket',
            'title': news[0]['title'],
            'text': news_text,
            'tags': [],
            'tickers': [],
            'companies': companies,
            'links': [news[0]['link']],
            'datetime': simple_date_convert(news[0]['date'])
        }

        if json_news != old_json_news:
            print(json_news)
            old_json_news = json_news
            send_json(json.dumps(json_news))

        time.sleep(1)

    connection.close()