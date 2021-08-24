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
                df[f'ch:{k}+{c}atr'] = df[k] + df['ATR'] * c
                df[f'ch:{k}-{c}atr'] = df[k] - df['ATR'] * c

    return df

def add_AD(df):
    '''Accumulation/ Distribution
    '''
    df['A/D'] = (
        df['Volume'] * (df['Close'] - df['Open'])/ (df['High'] - df['Low'])
        ).fillna(0).cumsum()
    return df

def add_OBV(df):
    '''On Balance Volume
    ref: https://stackoverflow.com/a/66827219/14285096
    '''
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV'] = obv
    return df

def add_RSI(df, n = 14, price_col = 'Close'):
    '''RSI
    ref: https://stackoverflow.com/a/57037866/14285096
    '''
    def rma(x, n, y0): # Running Moving Average
        a = (n-1) / n
        ak = a**np.arange(len(x)-1, -1, -1)
        return np.r_[np.full(n, np.nan), y0, np.cumsum(ak * x) / ak / n + y0 * a**np.arange(1, len(x)+1)]

    delta = df[price_col].diff()
    gains = delta.mask(delta < 0, 0.0)
    losses = -delta.mask(delta > 0, -0.0)
    avg_gain = rma(gains[n+1:].to_numpy(), n, np.nansum(gains.to_numpy()[:n+1])/n)
    avg_loss = rma(losses[n+1:].to_numpy(), n, np.nansum(losses.to_numpy()[:n+1])/n)
    df['RS'] = avg_gain / avg_loss
    df[f'RSI'] = 100 - (100 / (1 + df['RS']))
    return df

def add_ADX(df, period: int):
    """Computes the ADX indicator.
    source: https://stackoverflow.com/a/64946213/14285096
    """
    alpha = 1/period
    assert 'ATR' in df.columns, f"ATR must be computed before computing ADX"
    # +-DX
    df['H-pH'] = df['High'] - df['High'].shift(1)
    df['pL-L'] = df['Low'].shift(1) - df['Low']
    df['+DX'] = np.where(
        (df['H-pH'] > df['pL-L']) & (df['H-pH']>0),
        df['H-pH'],
        0.0
    )
    df['-DX'] = np.where(
        (df['H-pH'] < df['pL-L']) & (df['pL-L']>0),
        df['pL-L'],
        0.0
    )
    del df['H-pH'], df['pL-L']

    # +- DMI
    df['S+DM'] = df['+DX'].ewm(alpha=alpha, adjust=False).mean()
    df['S-DM'] = df['-DX'].ewm(alpha=alpha, adjust=False).mean()
    df['+DMI'] = (df['S+DM']/df['ATR'])*100
    df['-DMI'] = (df['S-DM']/df['ATR'])*100
    del df['S+DM'], df['S-DM']

    # ADX
    df['DX'] = (np.abs(df['+DMI'] - df['-DMI'])/(df['+DMI'] + df['-DMI']))*100
    df['ADX'] = df['DX'].ewm(alpha=alpha, adjust=False).mean()
    del df['DX'],df['-DX'], df['+DX']#, df['+DMI'], df['-DMI']

    return df

def add_Impulse(df, ema_name, MACD_Hist_name = "MACD_histogram"):
    ''' Elder's Impluse system
    positive sloping ema and positive sloping MACD Hist are green
    negative sloping ema and negative sloping MACD Hist are red
    all other bars are blue
    '''
    assert ema_name in df.columns, f'{ema_name} not found in input df'
    assert MACD_Hist_name in df.columns, f'{MACD_Hist_name} missing in df'

    ema_diff = df[ema_name] - df[ema_name].shift()
    macd_hist_diff = df[MACD_Hist_name] - df[MACD_Hist_name].shift()
    df['impulse'] = np.array([1 if d_ema > 0 and d_macdh > 0 else
                            -1 if d_ema < 0 and d_macdh < 0 else 0
                            for d_ema, d_macdh in zip(ema_diff, macd_hist_diff)]
                    )
    return df

def get_avg_penetration(df, price_col = 'Low', fair_col = 'ewa_11',
        num_of_bars = 22, get_dict = True):
    ''' Calculate the average pentration below the fair
    Args:
        price_col: to check when this col is below the fair
    '''
    assert price_col in df.columns, f'{price_col} not found in input df'
    assert fair_col in df.columns, f'{fair_col} not found in input df'
    assert len(df) >= num_of_bars, f'df only has {len(df)} bars'

    df_ = df.copy()[-num_of_bars:]
    has_penetrate = df_[price_col]< df_[fair_col]
    df_['penetration'] = df_[fair_col] - df_[price_col]
    expected_fair = df_[fair_col][-1] + (df_[fair_col][-1] - df_[fair_col][-2])

    return {'avg': df_[has_penetrate]['penetration'].mean(),
        'stdv': df_[has_penetrate]['penetration'].std(),
        'count': sum(has_penetrate),
        'expected_fair': expected_fair,
        'buy_target': expected_fair - df_[has_penetrate]['penetration'].mean()
        } if get_dict else df_[has_penetrate]['penetration'].mean()


def add_peaks(df, date_col = None, order = 3):
    ''' local minima & maxima detection
    Args:
        order: how many points on each side to use for the comparison
    ref: https://eddwardo.github.io/posts/finding-local-extreams-in-pandas-time-series/
    '''
    pass
