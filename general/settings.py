from datetime import datetime
import pytz

DAYS_TO_EXPIRE = 1
TIMEZONE = pytz.timezone('Europe/Moscow')

def in_period(target):
    return (datetime.now(tz=TIMEZONE) - target).days < DAYS_TO_EXPIRE