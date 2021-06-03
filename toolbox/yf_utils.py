import yfinance as yf

# for caching yfinance results
import requests_cache
session = requests_cache.CachedSession('yfinance.cache')
session.headers['User-agent'] = 'my-program/1.0'

def valid_stock(stock_obj):
    '''
    Check if a stock_obj created by yf.Ticker has data
    '''
    try:
        stock_obj.info
        return True
    except:
        return False

def get_stocks_data(tickers, session = session,
        yf_download_params = {"period": '1y', "group_by": "column" , "interval": "1d"}
        ):
    '''
    Return data using yf.download method
    Args:
        yf_download_params: passed straight to yf.download, see https://github.com/ranaroussi/yfinance
    '''
    prices_df = yf.download(tickers = tickers, **yf_download_params)
    returns_df = prices_df['Adj Close'].pct_change()[1:]
    col_with_returns = [col for col in returns_df.columns
                        if returns_df[col].isna().sum() < len(returns_df)
                        ]
    returns_df = returns_df[col_with_returns].dropna()
    return {
        'prices': prices_df,
        'returns': returns_df
    }
