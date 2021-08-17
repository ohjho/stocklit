import yfinance as yf
import pandas as pd
from tqdm import tqdm

# TODO: for caching yfinance results
# this is an open issue for the Ticker method: https://github.com/ranaroussi/yfinance/issues/677
import requests_cache
SESH = requests_cache.CachedSession('yfinance.cache')
SESH.headers['User-agent'] = 'my-program/1.0'

def valid_stock(stock_obj):
    '''
    Check if a stock_obj created by yf.Ticker has data
    '''
    try:
        stock_obj.info
        return True
    except:
        return False

def tickers_parser(tickers, return_list = False, max_items = None):
    '''
    parse a string of space separated tickers with special handling for HK tickers
    Args:
        return_list: return a list object if true else a space separated string
        max_items: maximum number of items to return if given otherwise return all
    '''
    if tickers:
        l_tickers = tickers.split()
        l_tickers = [ ticker.split('.')[0].zfill(4) + '.HK' if '.HK' in ticker.upper() else ticker
            for ticker in l_tickers ]

        l_tickers = l_tickers[:max_items] if max_items else l_tickers
        l_tickers = [t.upper() for t in l_tickers]

        return l_tickers if return_list else " ".join(l_tickers)
    else:
        return None

def get_stocks_data(tickers, session = SESH,
        yf_download_params = {"period": '1y', "group_by": "column" , "interval": "1d"}
        ):
    '''
    Return data using yf.download method
    See this issue with rate limiter: https://github.com/ranaroussi/yfinance/issues/602
    Args:
        yf_download_params: passed straight to yf.download, see https://github.com/ranaroussi/yfinance
    '''
    prices_df = yf.download(tickers = tickers, session = SESH, **yf_download_params)
    returns_df = prices_df['Adj Close'].pct_change()[1:]
    if len(tickers.split())> 1:
        col_with_returns = [col for col in returns_df.columns
                            if returns_df[col].isna().sum() < len(returns_df)
                            ]
        returns_df = returns_df[col_with_returns].dropna()
    else:
        returns_df = pd.DataFrame({tickers :returns_df}, index = returns_df.index)

    return {
        'prices': prices_df,
        'returns': returns_df
    }

def get_dfs_by_tickers(df):
    '''
    return a dictionary of {ticker:df,...}
    '''
    assert isinstance(df.columns, pd.MultiIndex), "this function only works with DF with MultiIndex Columns"
    data_col = set([c[0] for c in df.columns])
    tickers = set([c[1] for c in df.columns])

    l_DFs = [df[[(c,t) for c in data_col]] for t in tickers]
    for df in l_DFs:
        df.columns = df.columns.droplevel(1)
    return {t:df for t, df in zip(tickers, l_DFs)}

def get_stocks_obj(tickers, session = SESH, tqdm_func = tqdm):
    tickers_obj = yf.Tickers(tickers)
    return tickers_obj.tickers
    # return [yf.Ticker(t)
    #     for t in tqdm_func(tickers.split(), desc = "Creating Ticker Object")
    #     ]

def get_stocks_info(tickers, tqdm_func = tqdm):
    '''
    returns a list of dictionary for each ticker
    '''
    tickers_obj = get_stocks_obj(tickers)
    # print(tickers_obj)
    # tickers_obj : dict {'ticker': yfinance.Ticker object,...}

    # TODO: add try-except & retry
    results = [ {k: v for k, v in t.info.items()}
        for t in tqdm_func(tickers_obj.values(), desc = "Getting stocks info")
    ]
    return results

def df_to_weekly(df_daily):
    '''
    take a daily DF and convert it to weekly DF
    '''
    pass
