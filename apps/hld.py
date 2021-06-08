import os, sys, json, requests
import streamlit as st
import pandas as pd

STOCK_UNIVERSE = [
    {'name': 'S&P 500', 'index': '^GSPC', 'reference_security': 'SPY'},
    {'name': 'Hang Seng Index', 'index': '^HSI', 'reference_security': '2800.HK'}
]

@st.cache
def get_index_members(index_name):
    '''
    return a list of yf tickers for the given index
    '''
    if index_name == '^GSPC':
        # ref: https://medium.com/wealthy-bytes/5-lines-of-python-to-automate-getting-the-s-p-500-95a632e5e567
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = tables[0]
        return df['Symbol'].tolist()
    elif index_name == '^HSI':
        # see https://tcoil.info/build-simple-stock-trading-bot-advisor-in-python/
        # tables = pd.read_html('https://finance.yahoo.com/quote/%5EHSI/components/')
        # df = tables[0]
        # return df['Symbol'].tolist()

        # see https://medium.com/financial-data-analysis/step-1-web-scraping-hong-kong-hsi-stock-price-7d8606c07c57
        tables = pd.read_html(requests.get('http://www.etnet.com.hk/www/eng/stocks/indexes_detail.php?subtype=HSI',
                                       headers={'User-agent': 'Mozilla/5.0'}).text,
                                       header= 0)
        df = tables[2]
        return [str(s).zfill(4)+'.HK' for s in df['Code'].tolist()]

@st.cache
def get_etf_holdings(etf_ticker, parse = False):
    '''
    prototype function, doesn't work yet
    '''
    # need chromium webdriver
    # see: https://medium.com/hackernoon/python-notebook-research-to-replicate-etf-using-free-data-ca9f88eb7349
    ref_url = f'https://www.barchart.com/etfs-funds/quotes/{etf_ticker}/constituents?page=all'
    # tables = pd.read_html(ref_url, header = {'User-Agent': 'Mozilla/5.0'})
    tables = pd.read_html(requests.get(ref_url,
                                   headers={'User-agent': 'Mozilla/5.0'}).text,
                      attrs={"class":"constituents"} if parse else None)
    print(len(tables))
    df = tables[2]
    return df
