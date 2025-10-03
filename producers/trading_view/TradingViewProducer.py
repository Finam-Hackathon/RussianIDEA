from bs4 import BeautifulSoup
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from requests_html import HTMLSession
import os

from general.Settings import TIMEZONE, in_period

SESSION_ID = os.getenv('SESSION_ID')
SESSION_ID_SIGN = os.getenv('SESSION_ID_SIGN')
s = HTMLSession()
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
}
options = Options()
options.add_argument("--headless")
browser = webdriver.Chrome(options=options)
browser.get('https://ru.tradingview.com/news-flow/?market=bond,economic,etf,forex,futures,index,stock&market_country=entire_world&provider=reuters,rbc')
last_index = -1
scroll_view = browser.find_element('xpath', "//div[@class='container-KjK7wgQo table-t6J_7lf0']")
scrolled = 0
running = True
while running:
    links = BeautifulSoup(browser.page_source, 'html.parser').find_all('a')
    for link in links:
        if link.get('data-index') is None:
            continue
        data_index = int(link.get('data-index'))
        if data_index <= last_index:
            continue
        if data_index == 10:
            running = False
            break
        last_index = data_index
        page_link = link.get('href')
        r = s.get(f'https://ru.tradingview.com{page_link}', cookies={
            "sessionid": SESSION_ID,
            "sessionid_sign": SESSION_ID_SIGN
        })
        r.html.render()
        page = BeautifulSoup(r.content, 'html.parser')
        news_datetime = datetime.fromtimestamp(int(page.find("time-format").get('timestamp')) / 1000, tz=TIMEZONE)
        if not in_period(news_datetime):
            running = False
            break
        title = page.find("h1", {"data-qa-id": "news-description-title"}).text
        tickers_view = page.find("div", ["symbolsContainer-cBh_FN2P", "logosContainer-cwMMKgmm"])
        text = page.find("div", ["body-KX2tCBZq", "body-pIO_GYwT", "content-pIO_GYwT"]).text
        if tickers_view is not None:
            text = text.strip(tickers_view.text).strip()
        tickers = list(map(lambda x: x.text, tickers_view.find_all('span', ['description-cBh_FN2P']))) if tickers_view is not None else []
        tags_view = page.find("div", {"class": "rowTagsDefault-TeKDWl75"})
        tags = list(map(lambda x: x.text, tags_view.find_all('a')))
        print({
            'producer': 'TradingView',
            'title': title,
            'text': text,
            'tags': tags,
            'tickers': tickers,
            'link': f'https://ru.tradingview.com{page_link}',
            'datetime': news_datetime.strftime("%d.%m.%Y %H:%M")
        })
    scrolled += 100
    browser.execute_script("arguments[0].scrollTop = arguments[1]", scroll_view, scrolled)
