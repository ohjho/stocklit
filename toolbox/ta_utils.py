###############################################################################
# Inspiration from quantmod (https://github.com/jackluo/py-quantmod)
# & this kaggle notebook (https://www.kaggle.com/mtszkw/technical-indicators-for-trading-stocks)
#
# all functions accept input df from yfinance with columns:
#  ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
###############################################################################
import os, sys
import numpy as np
import pandas as pd

def add_moving_average(df, period:int, type: str = "ema", price_col = "Adj Close", vol_col = "Volume"):
    '''
    add a new column with moving average
    args:
        period: number of bars (row) in the df
        type: ema, sma, vwap
        price_col: name of the column to apply the calculation on
    '''
    assert price_col in df.columns, f'add_moving_average: {price_col} not found in df provided.'
    ma_dict = {
        'ema': df[price_col].ewm(period).mean().shift(),
        'sma': df[price_col].rolling(period).mean().shift(),
        'vwap': (df[price_col] * df[vol_col]).rolling(period).sum().shift() / df[vol_col].rolling(period).sum().shift()
    }
    assert type in ma_dict.keys(), f'add_moving_average: type {type} is not one of {ma_dict.keys()}'

    new_col_name = f'{type}_{period}'
    df[new_col_name] = ma_dict[type]
    return df

def add_MACD(df, fast:int= 12, slow:int= 26, signal:int = 9, price_col = "Adj Close"):
    ema_fast = df[price_col].ewm(span = fast, min_periods = fast).mean()
    ema_slow = df[price_col].ewm(span = slow, min_periods = slow).mean()
    MACD = ema_fast - ema_slow
    MACD_signal = MACD.ewm(span = signal, min_periods = signal).mean()
    df['MACD_histogram'] = MACD - MACD_signal
    df['MACD'] = MACD
    df['MACD_signal'] = MACD_signal
    return df

def add_ATR(df, period: int = 13, use_ema = False,
        channel_dict = {'ema_22': [1,2,3]}
    ):
    '''
    Average True Range
    Args:
        channel_dict:
        use_ema: use exponential moving average instead of simple
        channel_dict: name of price and list of ATR multiples to apply
    ref: https://www.learnpythonwithrune.org/calculate-the-average-true-range-atr-easy-with-pandas-dataframes/
    ref: https://stackoverflow.com/questions/40256338/calculating-average-true-range-atr-on-ohlc-data-with-python
    '''
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['TR'] = true_range
    df['ATR'] = true_range.ewm(span = period).mean() if use_ema else \
                true_range.rolling(period).mean()

    if channel_dict:
        for k, l_channels in channel_dict.items():
            assert k in df.columns, f'add_ATR: {k} is not found in input df.'
            for c in l_channels:
                df['{k}+{c}atr'] = df[k] + df['ATR'] * c
                df['{k}-{c}atr'] = df[k] - df['ATR'] * c

    return df

def add_AD(df):
    '''Advance/ Decline
    '''
    df['A/D'] = df['Volume'] * (df['Close'] - df['Open'])/ (df['High'] - df['Low'])
    return df

def add_OBV(df):
    '''On Balance Volume
    ref: https://stackoverflow.com/a/66827219/14285096
    '''
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV'] = obv
    return df
