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
    positive sloping ema and positive sloping MACD Hist are green (1)
    negative sloping ema and negative sloping MACD Hist are red (-1)
    all other bars are blue (0)
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

def add_avg_penetration(df, hilo_col_names = ('High','Low'), fair_col = 'ema_11',
        num_of_bars = 30, use_ema = False, ignore_zero = True, coef = 1,
        get_df = True, debug = False):
    ''' Calculate the average pentration below & above the fair and
        return buy and sell SafeZones
    Args:
        use_ema: use exponential moving average to compute average penetration
        ignore_zero: ignore days without penetration
        coef: applied to average penetration to create SafeZones
    '''
    hi, lo = hilo_col_names
    assert hi in df.columns, f'{hi} not found in input df'
    assert lo in df.columns, f'{hi} not found in input df'
    assert fair_col in df.columns, f'{fair_col} not found in input df'
    assert len(df) >= num_of_bars, f'df only has {len(df)} bars'

    df_ = df.copy()
    df_['up'] = (df_[hi] - df_[fair_col]).clip(lower = 0)   # upper penetration
    df_['lp'] = (df_[fair_col] -df_[lo]).clip(lower = 0)    # lower penetration

    for pen in ['up', 'lp']:
        if use_ema:
            # ref: https://stackoverflow.com/a/64969439/14285096
            df_[f'avg_{pen}'] = df_[pen].replace(0, np.nan).ewm(num_of_bars, ignore_na = True).mean().shift() \
                                if ignore_zero else \
                                df_[pen].ewm(num_of_bars, ignore_na = True).mean().shift()
            df_[f'std_{pen}'] = df_[pen].ewm(num_of_bars, ignore_na = True).std().shift() \
                                if ignore_zero else \
                                df_[pen].ewm(num_of_bars, ignore_na = True).std().shift()
        else: # simple average
            df_[f'avg_{pen}'] = df_[pen].rolling(num_of_bars).apply(lambda x: x[x>0].mean()).shift() \
                                if ignore_zero else \
                                df_[pen].rolling(num_of_bars).mean().shift()
            df_[f'std_{pen}'] = df_[pen].rolling(num_of_bars).apply(lambda x: x[x>0].std()).shift() \
                                if ignore_zero else \
                                df_[pen].rolling(num_of_bars).std().shift()
        df_[f'count_{pen}'] = df_[pen].rolling(num_of_bars).apply(lambda x: x[x>0].count()).shift()

    df_['buy_safezone'] = df_[fair_col] - (df_['avg_lp'] * coef)
    df_['sell_safezone'] = df_[fair_col] + (df_['avg_up'] * coef)

    # expected_fair = df_[fair_col][-1] + (df_[fair_col][-1] - df_[fair_col][-2])
    if not debug and get_df:
        del df_['up'], df_['lp']
        for p in ['up', 'lp']:
            del df_[f'avg_{p}'], df_[f'std_{p}'], df_[f'count_{p}']

    return df_ if get_df else {
        'avg_upper_penetration': df_['avg_up'][-1],
        'stdv_upper_penetration': df_['std_up'][-1],
        'count_upper_penetration': df_['count_up'][-1],
        'avg_lower_penetration': df_['avg_lp'][-1],
        'stdv_lower_penetration': df_['std_lp'][-1],
        'count_lower_penetration': df_['count_lp'][-1],
        # 'expected_fair': expected_fair,
        # 'buy_target_t+1': expected_fair - df_['avg_lp'][-1]
    }

def market_classification(df, period, debug = False,
        class_names_map = {0: 'rangebound', 1: 'trending-up', -1: 'trending-down'} ):
    ''' detect trending-up, trending-down, or rangebound
    '''
    df['period_high'] = df['High'].rolling(period).max().shift()
    df['period_low'] = df['Low'].rolling(period).min().shift()

    highs = [round(p,2) for p in df['period_high'][-period:].unique().tolist()]
    if debug:
        print(f'{period} bars highs: {highs}')
    hh = sorted(highs) == highs
    lh = sorted(highs, reverse = True) == highs

    lows = [round(p,2) for p in df['period_low'][-period:].unique().tolist()]
    if debug:
        print(f'{period} bars lows: {lows}')
    ll = sorted(lows, reverse = True) == lows
    hl = sorted(lows) == lows

    mkt_cls = -1 if ll and lh else 0
    mkt_cls = 1 if hh and hl else mkt_cls
    return class_names_map[mkt_cls] if class_names_map else mkt_cls

def detect_kangaroo_tails(df, col_name = 'kangaroo_tails',
        atr_threshold = 2, period = 13, hilo_col_name_tup = ('High','Low'),
        tail_type = 0, debug = False
    ):
    ''' add a new column to the df when 1 indicates a kangaroo tail
    on the previous bar
    see: https://www.bwts.com.au/download/lc-charting-and-technical-analysis/41-kangaroo-tail-pattern.pdf
    Args:
        tail_type: 1 for upward pointing tails, -1 for downward pointing, 0 for both
    '''
    assert 'ATR' in df.columns, f"ATR calculation is required before this function is ran"
    hi_col, low_col = hilo_col_name_tup
    df['is_new_low'] = (df[low_col] < df[low_col].rolling(period).min().shift())
    df['is_new_hi'] = (df[hi_col] > df[hi_col].rolling(period).max().shift())
    df['pct_of_ATR'] = (df['TR'] / df['ATR'].shift())
    hilo = df[hi_col] - df[low_col] # can use TR because it includes change from prev close
    pre_req_condition = (
        (df['pct_of_ATR'].shift() > atr_threshold) & # previous bar has to be tall
        ((hilo <= df['ATR']) & (hilo.shift(2) <= df['ATR'])) &  # adjusant bodies are normal
        (~df['is_new_hi'] & ~df['is_new_low']) # today is a normal day
        )
    downtail_condition = (
        (df['Low']> df['Low'].shift()) &
        ((df['High'].shift(2) > df['High'].shift()) &
            (df['Low'].shift(2) <= df['High'].shift())
            ) & # previous high is in t-2 range
        ((df['High'] > df['High'].shift()) &
            (df['Low'] <= df['High'].shift())
            ) # previous high is in today's range
        )
    uptail_condition = (
        (df['High'].shift() > df['High']) &
        ((df['High'].shift(2) >= df['Low'].shift()) &
            (df['Low'].shift(2) < df['Low'].shift())
            ) & # previous high is in t-2 range
        ((df['High'] >= df['Low'].shift()) &
            (df['Low'] < df['Low'].shift())
            ) # previous high is in today's range
        )
    # for i, row in df.reset_index().iterrows():
    #     if i > 0:
    #         if is_new_hi[i-1] or is_new_low[i-1]:
    #             if pct_of_ATR[i-1]> atr_threshold and \
    #             pct_of_ATR[i]< atr_threshold and pct_of_ATR[i-2]< atr_threshold:
    #                 if not is_new_hi[i] and not is_new_low[i]:
    #                     df[col_name] = 1
    if not debug:
        del df['is_new_hi'], df['is_new_low'], df['pct_of_ATR']

    tail_con = (pre_req_condition & downtail_condition) if tail_type == -1 else \
                (pre_req_condition & uptail_condition)
    tail_con = (pre_req_condition & downtail_condition & uptail_condition) \
                if tail_type == 0 else tail_con
    df[col_name] = tail_con
    return df

def cluster_near_by_sign(data, n= 1):
    ''' cluster neigboring points by sign
    ref: https://stackoverflow.com/a/33812257/14285096
    '''
    from itertools import groupby
    all_clusters = [list(v) for k, v in groupby(data, lambda x: x<0)]
    return all_clusters[-n:] if n else all_clusters

def detect_macd_divergence(df, period = 66, threshold = 1, debug = False):
    ''' Detect MACD Divergence
    Args:
        period = number of bars (should be around 3M)
        threshold = current price less than previous low * threshold
    '''
    macd_h = 'MACD_histogram'
    assert macd_h in df.columns, f"Please run add_MACD() first"
    do_impulse = 'impulse' in df.columns
    # Computation of recent period is the problem here
    # recent_period = int(period/3) # power of 5? maybe 1/3 better?

    # Bullish Divergence
    current_clusters_size = df[macd_h].rolling(period).apply(
                            lambda x: len(cluster_near_by_sign(x, n =1)[0])
                            ).shift()
    lows_ll = df['Low'].tolist()
    period_lows = [lows_ll[i-period: i] if i>= period else []
                        for i, v in enumerate(lows_ll)
                        ]
    recent_low = [ min(lows[-size:]) if size and len(lows)>0 else np.nan
        for lows, size in zip( period_lows,
            current_clusters_size.fillna(0).astype('int').tolist()
            )
        ]

    # is_new_low = ((df['Low'].rolling(period).min().shift()/df['Low']) >= threshold)
    # is_new_low = ((df['Low'].rolling(period).min().shift()/recent_low) >= threshold)

    # using recent_low is better than just lowing at Low in that it capture false downside breakouts
    is_new_low = ((df['Low'].rolling(period).min().shift()/recent_low) >= threshold)
    is_macd_low = (df[macd_h] < df[macd_h].rolling(period).min().shift())
    # recent_macd_low = df[macd_h].rolling(recent_period).min().shift()
    recent_macd_low = df[macd_h].rolling(period).apply(
                        lambda x: min(cluster_near_by_sign(x, n =1)[0])
                        ).shift()
    recent_macd_low_is_low = (recent_macd_low <= df[macd_h].rolling(period).min().shift())
    recent_macd_low_over_threshold = (
        recent_macd_low/ df[macd_h].rolling(period).min().shift() >= threshold
    )

    # number_of_macdH_clusters removed the need for this
    # recent_macd_high = df[macd_h].rolling(recent_period).max().shift()
    number_of_macdH_clusters = df[macd_h].rolling(period).apply(
                                lambda x: len(cluster_near_by_sign(x, n = None))
                                ).shift()
    impulse_check = (df['impulse'] != -1) if do_impulse else True

    df['MACD_Divergence'] = (is_new_low & ~is_macd_low &
                            (df[macd_h]< 0) &
                            ~recent_macd_low_is_low &
                            ~recent_macd_low_over_threshold &
                            # (recent_macd_high >= 0) &
                            (number_of_macdH_clusters >= 2 ) &
                            impulse_check
                            )
    if debug:
        df[f'{period}_bars_low'] = df['Low'].rolling(period).min().shift()
        df['recent_cluster_size'] = current_clusters_size
        df['recent_low'] = recent_low
        df['is_new_low'] = is_new_low
        df['MACDh_period_low'] = df[macd_h].rolling(period).min().shift()
        df['MACDh_recent_low'] = recent_macd_low
        df['MACDh_recent_low_over_thresh'] = recent_macd_low_over_threshold
        df['MACDh_is_new_low'] = is_macd_low
        df['MACDh_culsters_count'] = number_of_macdH_clusters
        df['MACDh_impulse_check'] =impulse_check
    return df

def add_peaks(df, date_col = None, order = 3):
    ''' local minima & maxima detection
    Args:
        order: how many points on each side to use for the comparison
    ref: https://eddwardo.github.io/posts/finding-local-extreams-in-pandas-time-series/
    '''
    pass
