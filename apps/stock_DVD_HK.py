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

def get_dvd(tickers, ex_date_after = None, aggregate = True,
        tqdm_func = stqdm, debug_mode = False):
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

            # Remove Things without exdate or Div
            filter_con = ((df['Ex-date']=='--') | df['Div'].isna())
            if len(df[filter_con]) > 0 and debug_mode:
                print(f'--- get_dvd: the following records will not be shown ---')
                print(df[filter_con])
            df = df[~filter_con]
            l_df.append(df)

    if len(l_df)==0:
        return None
    df_dvd = pd.concat(l_df)

    # dtype conversion
    df_dvd['Ex-date'] = pd.to_datetime(df_dvd['Ex-date'])
    df_dvd['Div'] = df_dvd['Div'].astype(float)

    if ex_date_after:
        # feature engineering
        d2ex = [(d- str_to_date(ex_date_after)).days for d in df_dvd['Ex-date'].tolist()]
        df_dvd.insert(loc = 4, column = 'Days-to-ex', value = d2ex)

        # sort and return
        df_dvd = df_dvd.sort_values(by = ['Ex-date']).reset_index().drop(columns=['index'])
    return df_dvd if aggregate else div_dict

def visualize_dvd_df(df_dvd):
    # Div/ATR histogram
    fig = px.histogram(df_dvd, x = 'div/ATR', color = 'ticker',
            title = 'div/ATR Histogram')
    show_plotly(fig)

    # Days vs Div/ATR scatter
    if 'Days-to-ex' in df_dvd.columns:
        fig = px.scatter(df_dvd, x = 'Days-to-ex', y = 'div/ATR', color = 'ticker',
                title = f'Days-to-ex vs div/ATR')#, hover_data = [''])
        show_plotly(fig)

    if 'TR_ex' in df_dvd.columns:
        fig = px.scatter(df_dvd, x = 'TR_ex', y = 'Div', color = 'ticker',
                title = f'Div vs True Range (on ex-date)')#, hover_data = [''])
        show_plotly(fig)

    if 'capture_r_multiplier' in df_dvd.columns:
        df_p = df_dvd[df_dvd['capture_risk']>0]
        fig = px.histogram(df_p, x = 'capture_r_multiplier', color = 'ticker',
                title = 'Dividend Capture R-Multiplier Distribution')
        show_plotly(fig)

def show_upcoming_div(df, st_asset, timeframe_params, atr_period = 22):
    with st_asset:
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
            ohlc_dict = { t: add_ATR(df.copy(), period = atr_period, use_ema = True, channel_dict = None)
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

def get_df_value_by_date(df, target_col, date, debug = False):
    if date in df.index:
        return df.loc[date][target_col]
    else:
        if debug:
            print(f"get_df_value_by_date: {date} not in df's index")
        return None

def show_past_div(df, st_asset, timeframe_params, atr_period):
    with st_asset:
        if not isinstance(df, pd.DataFrame):
            st.warning('no dividends found')
            return None
        div_tickers = df['ticker'].unique().tolist()
        df = df[df['Ex-date']< pd.Timestamp(timeframe_params['end_date'])].copy()

        # Get Price Data
        ohlc_dict = get_ohlc_data(div_tickers, start_date = timeframe_params['data_start_date'],
                            end_date = timeframe_params['end_date'],
                            interval = timeframe_params["interval"]
                        )
        if 'ATR' not in ohlc_dict[div_tickers[0]].columns:
            ohlc_dict = { t: add_ATR(df.copy(), period = atr_period, use_ema = True, channel_dict = None)
                            for t, df in ohlc_dict.items()
                            }
        # add ATR, TR
        atr = df.apply(
                    lambda row: get_df_value_by_date(
                        df = ohlc_dict[row['ticker']], target_col = 'ATR',
                        date = (BusinessDate(row['Ex-date'])-'1b').to_date()
                        ), axis = 1
                )
        tr = df.apply(
                    lambda row: get_df_value_by_date(
                        df = ohlc_dict[row['ticker']], target_col = 'TR',
                        date = row['Ex-date'].date()
                        ), axis = 1
                )
        df.insert(loc = 4, column = 'ATR_ex-1', value = atr)
        df.insert(loc = 5, column = 'TR_ex', value = tr)
        df.insert(loc = 4, column = 'div/ATR', value = df['Div']/ atr)

        # User-Input on Capture
        if st.checkbox('test dividend capture'):
            l_col, r_col = st.beta_columns(2)
            atr_factor = l_col.number_input('number of ATR to risk', value = 2.0)
            df['capture_risk'] = df.apply(
                lambda row: atr_factor * row['ATR_ex-1'] if row['div/ATR']> atr_factor else 0,
                axis = 1)
            df['capture_reward'] = df.apply(
                lambda row: row['Div']-row['TR_ex']if row['capture_risk']>0 else 0,
                axis = 1)
            df['capture_r_multiplier'] = df.apply(
                lambda row: row['capture_reward']/ row['capture_risk'] if row['capture_risk']>0 else 0,
                axis = 1)

            results_dict = {
                'trades' : len(df[df["capture_risk"]>0]),
                'wins':  len(df[df['capture_reward']>0]),
                'expectancy': df[df["capture_risk"]>0]['capture_r_multiplier'].mean()
                }
            results_dict['win_rate'] = results_dict['wins']/results_dict['trades']
            r_col.write(results_dict)
        st.write(df)

        visualize_dvd_df(df.dropna(subset= ['Div','ATR_ex-1']))

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
                            st_asset = st.sidebar.beta_expander("Timeframe"),
                            default_tenor = '50y'
                        )
    if tickers:
        # - div history with ex-date true range vs div amount/ true range
        df = get_dvd(tickers, ex_date_after = timeframe_params['end_date'],
                aggregate = True)
        show_upcoming_div(df, st_asset = st.beta_expander("View Upcoming Dividend", expanded = True),
            timeframe_params= timeframe_params, atr_period= 22)

        df_all = get_dvd(tickers, aggregate = True)
        show_past_div(df_all, st_asset = st.beta_expander("View Pass Dividend"),
            timeframe_params = timeframe_params, atr_period= 22)


if __name__ == '__main__':
    Main()
