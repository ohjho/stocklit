import os, sys, json, datetime
import streamlit as st
import pandas as pd
import plotly.express as px
from businessdate import BusinessDate

#Paths
cwdir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, os.path.join(cwdir, "../"))
# from toolbox.st_utils import show_plotly, plotly_hist_draw_hline
from toolbox.yf_utils import tickers_parser, get_dfs_by_tickers
# from toolbox.plotly_utils import plotly_ohlc_chart
from toolbox.ta_utils import add_ATR
from apps.stock_returns import get_yf_data

def get_ATR_calc(df, period, use_ema, atr_multiplier = 2, var = 1000, price_col = 'Adj Close'):
    '''
    return a dictionary of various ATR calcuations
    Args:
        df: dataframe of prices from yfinance for a Single Stock
        var: Value At Risk
    '''
    data = add_ATR(df.copy(),  period = period, use_ema = use_ema, channel_dict = None)
    ATR = data['ATR'][-1]
    price = data[price_col][-1]
    shares = price / (atr_multiplier * ATR)
    return {'ATR': ATR,
        price_col : price,
        'ATR%Price': ATR/ price,
        'shares': shares,
        'position_size': shares * price
        }

def Main():
    with st.sidebar.beta_expander("ATR"):
        st.info(f'''
            #### Average True Range
            Compare risk across multiple stocks and determine your position sizing

            * data by [yfinance](https://github.com/ranaroussi/yfinance)
            * [definition](https://www.thebalance.com/how-average-true-range-atr-can-improve-trading-4154923)
            * [using ATR in position sizing](https://therobusttrader.com/how-to-use-atr-in-position-sizing/)
        ''')

    tickers = tickers_parser(st.text_input('enter stock ticker(s) [space separated]'), max_items = None)
    with st.sidebar.beta_expander('timeframe', expanded = False):
        today = datetime.date.today()
        end_date = st.date_input('Period End Date', value = today)
        if st.checkbox('pick start date'):
            start_date = st.date_input('Period Start Date', value = today - datetime.timedelta(days = 365))
        else:
            tenor = st.text_input('Period', value = '250b')
            start_date = (BusinessDate(end_date) - tenor).to_date()
            st.info(f'period start date: {start_date}')

        # TODO: allow manual handling of data_start_date
        data_start_date = (BusinessDate(start_date) - "1y").to_date()
        l_interval = ['1d','1m', '5m','1h','1wk']
        interval = st.selectbox('interval', options = l_interval)
        if interval.endswith(('m','h')):
            st.warning(f'intraday data cannot extend last 60 days')

    if tickers:
        with st.sidebar.beta_expander('ATR Configs', expanded = True):
            use_ema = st.checkbox('use exponential moving aveage')
            atr_period = st.number_input('ATR Period (number of bars)', value = 22)
            atr_multiplier = st.number_input('ATR multiplier (stop-loss size)', value = 2, step = 1)
            var = st.number_input('Value To Risk (max drawdown on one position)', value = 1000)

        data_dict = get_yf_data(tickers, start_date = data_start_date, end_date = end_date, interval = interval)
        data = data_dict['prices'].copy()
        df_return = data_dict['returns'].copy()

        with st.beta_expander('raw data'):
            st.subheader('Price Data')
            st.write(data)

        l_tickers = df_return.columns.tolist()
        if len(l_tickers) != len(tickers.split(' ')):
            st.warning(f'having trouble finding the right ticker?\nCheck it out first in `DESC` :point_left:')

        # l_DFs is a list of [ {ticker: df}, ...]
        df_dict = { l_tickers[0]: data} if len(l_tickers) ==1 else \
                get_dfs_by_tickers(data)
        results = [ {**{'ticker': t},
                    **get_ATR_calc(df, period = atr_period,
                        use_ema = use_ema, atr_multiplier = atr_multiplier,
                        var = var, price_col = 'Adj Close'
                        )}
            for t, df in df_dict.items() ]

        with st.beta_expander(f'ATR Results', expanded = True):
            st.write(pd.DataFrame(results))

if __name__ == '__main__':
    Main()
