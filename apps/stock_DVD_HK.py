import os, sys, json, datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from businessdate import BusinessDate
from stqdm import stqdm

#Paths
cwdir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, os.path.join(cwdir, "../"))
from toolbox.st_utils import show_plotly, plotly_hist_draw_hline, get_timeframe_params
from toolbox.yf_utils import tickers_parser, get_dfs_by_tickers
from toolbox.data_utils import str_to_date
from toolbox.hkex_utils import scrap_hk_stock_div
from toolbox.ta_utils import add_ATR
from apps.stock_members import get_index_tickers
from apps.stock_returns import get_yf_data

@st.cache
def get_ohlc_data(l_tickers, start_date, end_date, interval):
    ''' return a dictionary of tickers and ohlc df
    '''
    tickers = " ".join(l_tickers)
    data_dict = get_yf_data(tickers, start_date = start_date, end_date = end_date, interval = interval)
    data = data_dict['prices'].copy()
    df_dict = { l_tickers[0]: data} if len(l_tickers) ==1 else \
            get_dfs_by_tickers(data)
    return df_dict

def get_dvd(tickers, ex_date_after, aggregate = True, tqdm_func = stqdm):
    '''
    return a dictionary of the ticker and dividend dataframe
    Args:
        aggregate: if true, return one DF of all tickers' dividend data
    '''
    div_dict = {t: scrap_hk_stock_div(t, ex_date_after = ex_date_after)
        for t in tqdm_func(tickers, "loading dividend")}

    l_df = []
    for t, df in div_dict.items():
        if len(df)> 0:
            if 'ticker' not in df.columns:
                df.insert(loc = 0, column = 'ticker', value = t)
            l_df.append(df)

    if len(l_df)==0:
        return None
    df_dvd = pd.concat(l_df)

    # dtype convesion
    df_dvd['Ex-date'] = pd.to_datetime(df_dvd['Ex-date'])
    df_dvd['Div'] = df_dvd['Div'].astype(float)

    # feature engineering
    d2ex = [(d- str_to_date(ex_date_after)).days for d in df_dvd['Ex-date'].tolist()]
    df_dvd.insert(loc = 4, column = 'Days-to-ex', value = d2ex)

    # sort and return
    df_dvd = df_dvd.sort_values(by = ['Ex-date']).reset_index().drop(columns=['index'])
    return df_dvd if aggregate else div_dict

def visualize_dvd_df(df_dvd):
    # Div/ATR histogram
    fig = px.histogram(df_dvd, x = 'div/ATR', color = 'ticker',
            title = 'div/ATR Histogram', nbins = 10)
    show_plotly(fig)

    # Days vs Div/ATR scatter
    fig = px.scatter(df_dvd, x = 'Days-to-ex', y = 'div/ATR', color = 'ticker',
            title = f'Days-to-ex vs div/ATR')#, hover_data = [''])
    show_plotly(fig)

def Main():
    with st.sidebar.beta_expander("DVD_HK"):
        st.info(f'''
            Hong Kong Listed Stocks Upcoming Dividends
            ''')

    default_tickers = get_index_tickers(
                        st_asset = st.sidebar.beta_expander('Load an Index', expanded = True)
                        )
    tickers = tickers_parser(
                st.text_input('enter stock ticker(s) [space separated]',
                    value = default_tickers),
                return_list = True
                )
    timeframe_params = get_timeframe_params(
                            st_asset = st.sidebar.beta_expander("Timeframe")
                        )
    if tickers:
        df = get_dvd(tickers, ex_date_after = timeframe_params['end_date'],
                aggregate = True)
        if not isinstance(df, pd.DataFrame):
            st.warning('no upcoming dividends found')
            return None
        div_tickers = df['ticker'].tolist()

        # Get Price Data
        ohlc_dict = get_ohlc_data(div_tickers, start_date = timeframe_params['data_start_date'],
                            end_date = timeframe_params['end_date'],
                            interval = timeframe_params["interval"]
                        )
        if 'ATR' not in ohlc_dict[div_tickers[0]].columns:
            ohlc_dict = { t: add_ATR(df.copy(), period = 22, use_ema = True, channel_dict = None)
                            for t, df in ohlc_dict.items()
                            }

        # Update DVD df
        atr = df['ticker'].apply(
                    lambda x: ohlc_dict[x]['ATR'][-1]
                )
        df.insert(loc = 5, column = 'ATR', value = atr)
        df.insert(loc = 4, column = 'div/ATR', value = df['Div']/ atr)
        st.write(df)

        if st.checkbox('export tickers'):
            st.info(" ".join(div_tickers))

        visualize_dvd_df(df)

if __name__ == '__main__':
    Main()
