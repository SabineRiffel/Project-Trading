""" Scrape stock transactions from Senator periodic filings (resumable + ETA logging) """

from bs4 import BeautifulSoup
import logging
import pandas as pd
import requests
import time
import os
from datetime import datetime, timedelta

ROOT = 'https://efdsearch.senate.gov'
LANDING_PAGE_URL = f'{ROOT}/search/home/'
SEARCH_PAGE_URL = f'{ROOT}/search/'
REPORTS_URL = f'{ROOT}/search/report/data/'

BATCH_SIZE = 100
RATE_LIMIT_SECS = 2
PDF_PREFIX = '/search/view/paper/'
OUTPUT_CSV = 'senator_transactions_all.csv'
MAX_RETRIES = 3
TIMEOUT = 10

REPORT_COL_NAMES = [
    'tx_date',
    'file_date',
    'last_name',
    'first_name',
    'order_type',
    'ticker',
    'asset_name',
    'tx_amount',
    'link'
]

LOGGER = logging.getLogger(__name__)

def add_rate_limit(f):
    def wrapper(*args, **kwargs):
        time.sleep(RATE_LIMIT_SECS)
        return f(*args, **kwargs)
    return wrapper

def _csrf(client: requests.Session) -> str:
    landing_page_response = client.get(LANDING_PAGE_URL)
    assert landing_page_response.url == LANDING_PAGE_URL, "Failed to fetch filings landing page"
    landing_page = BeautifulSoup(landing_page_response.text, "html.parser")
    form_csrf = landing_page.find(attrs={'name': 'csrfmiddlewaretoken'})['value']

    client.post(LANDING_PAGE_URL,
                data={'csrfmiddlewaretoken': form_csrf, 'prohibition_agreement': '1'},
                headers={'Referer': LANDING_PAGE_URL})

    return client.cookies.get('csrftoken') or client.cookies.get('csrf')

def reports_api(client: requests.Session, start_date: str, end_date: str, token: str):
    data = {
        'start': '0',  # immer vom Anfang
        'length': str(BATCH_SIZE),
        'report_types': '[11]',
        'filer_types': '[]',
        'submitted_start_date': start_date,
        'submitted_end_date': end_date,
        'candidate_state': '',
        'senator_state': '',
        'office_id': '',
        'first_name': '',
        'last_name': '',
        'csrfmiddlewaretoken': token
    }
    LOGGER.info(f'Getting reports from {start_date} to {end_date}')

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.post(REPORTS_URL, data=data, headers={'Referer': SEARCH_PAGE_URL}, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.json()['data']
        except Exception as e:
            LOGGER.warning(f'Attempt {attempt+1} failed: {e}')
            time.sleep(2 ** attempt)
    raise RuntimeError(f'Failed to fetch reports from {start_date} to {end_date} after {MAX_RETRIES} attempts')

def _tbody_from_link(client: requests.Session, link: str):
    report_url = f'{ROOT}{link}'
    resp = client.get(report_url)
    if resp.url == LANDING_PAGE_URL:
        _csrf(client)
        resp = client.get(report_url)
    report = BeautifulSoup(resp.text, "html.parser")
    tbodies = report.find_all('tbody')
    return tbodies[0] if tbodies else None

def txs_for_report_all(client: requests.Session, row):
    first, last, _, link_html, date_received = row
    link_soup = BeautifulSoup(link_html, "html.parser")
    a_tag = link_soup.a
    link = a_tag.get('href') if a_tag else None

    if not link or link.startswith(PDF_PREFIX):
        return pd.DataFrame([{
            'tx_date': None,
            'file_date': date_received,
            'last_name': last,
            'first_name': first,
            'order_type': None,
            'ticker': None,
            'asset_name': None,
            'tx_amount': None,
            'link': f"{ROOT}{link}" if link else None
        }])

    tbody = _tbody_from_link(client, link)
    if not tbody:
        return pd.DataFrame([{
            'tx_date': None,
            'file_date': date_received,
            'last_name': last,
            'first_name': first,
            'order_type': None,
            'ticker': None,
            'asset_name': None,
            'tx_amount': None,
            'link': f"{ROOT}{link}"
        }])

    stocks = []
    for tr in tbody.find_all('tr'):
        cols = [c.get_text().strip() for c in tr.find_all('td')]
        if len(cols) < 8:
            continue
        tx_date, ticker, asset_name, asset_type, order_type, tx_amount = \
            cols[1], cols[3], cols[4], cols[5], cols[6], cols[7]
        if asset_type != 'Stock' and ticker.strip() in ('--', ''):
            continue
        stocks.append({
            'tx_date': tx_date,
            'file_date': date_received,
            'last_name': last,
            'first_name': first,
            'order_type': order_type,
            'ticker': ticker,
            'asset_name': asset_name,
            'tx_amount': tx_amount,
            'link': f"{ROOT}{link}"
        })
    return pd.DataFrame(stocks)

def main():
    LOGGER.info('Initializing client')
    client = requests.Session()
    client.get = add_rate_limit(client.get)
    client.post = add_rate_limit(client.post)

    token = _csrf(client)

    start = datetime(2012, 1, 1)
    end = datetime.today()

    # CSV vorbereiten
    if os.path.exists(OUTPUT_CSV):
        mode = 'a'
        header = False
    else:
        mode = 'w'
        header = True

    while start < end:
        month_end = (start.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        start_str = start.strftime("%m/%d/%Y 00:00:00")
        end_str = month_end.strftime("%m/%d/%Y 23:59:59")

        batch = reports_api(client, start_str, end_str, token)

        for r in batch:
            df = txs_for_report_all(client, r)
            df.to_csv(OUTPUT_CSV, mode=mode, header=header, index=False)
            header = False
            mode = 'a'

        start = month_end + timedelta(days=1)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s')
    main()
