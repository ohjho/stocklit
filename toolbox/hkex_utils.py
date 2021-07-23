import pandas as pd
import numpy as np
from openpyxl import load_workbook
from io import BytesIO
from urllib.request import urlopen

from functools import lru_cache, wraps
from datetime import datetime, timedelta

def timed_lru_cache(seconds: int, maxsize: int = 128):
    ''' add an expiry to the lru_cache
    ref: https://realpython.com/lru-cache-python/
    '''
    def wrapper_cache(func):
        func = lru_cache(maxsize=maxsize)(func)
        func.lifetime = timedelta(seconds=seconds)
        func.expiration = datetime.utcnow() + func.lifetime

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            if datetime.utcnow() >= func.expiration:
                func.cache_clear()
                func.expiration = datetime.utcnow() + func.lifetime
            return func(*args, **kwargs)

        return wrapped_func

    return wrapper_cache

@timed_lru_cache(seconds = 8*60**2) # Cache data for 8 hours
def get_hkex_securities_df(xlsx_url = 'https://www.hkex.com.hk/eng/services/trading/securities/securitieslists/ListOfSecurities.xlsx',
        convert_stock_code = True):
    '''return a df of HK lot sizes for all HK stocks
    Args:
        xlsx_url: url of the HKEX's list of securities
        convert_stock_code: convert stock code to yfinance tickers
    '''
    print(f'hkex_utils: initializing data from {xlsx_url}')
    wb_file_obj = urlopen(xlsx_url).read()
    wb = load_workbook(filename = BytesIO(wb_file_obj))
    ws = wb['ListOfSecurities']
    ws.delete_rows(0,2)
    data = ws.values
    df = pd.DataFrame(data, columns = next(data)[0:])
    df = df.dropna(how = 'all')
    if convert_stock_code:
        df['Stock Code'] = df['Stock Code'].apply( lambda x : str(int(x)).zfill(4) + '.HK')
    return df

def get_lot_size(ticker, df_sec = get_hkex_securities_df()):
    lot_size = df_sec[df_sec['Stock Code'] == ticker.upper()]['Board Lot']
    if len(lot_size) > 0:
        lot_size = lot_size.tolist()[0]
        # comma handling
        # TODO: the proper way https://www.delftstack.com/howto/python/how-to-convert-string-to-float-or-int/#commas-as-thousand-seperator-in-us-or-uk
        return int(lot_size.replace(',',''))
    else:
        return None
