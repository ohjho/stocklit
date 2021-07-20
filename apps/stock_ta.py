import os, sys, json, datetime
import streamlit as st
import pandas as pd
import plotly.express as px
from businessdate import BusinessDate

#Paths
cwdir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, os.path.join(cwdir, "../"))
from toolbox.st_utils import show_plotly, plotly_hist_draw_hline
from toolbox.yf_utils import tickers_parser
from toolbox.plotly_utils import plotly_ohlc_chart
from toolbox.ta_utils import add_moving_average, add_MACD, add_AD, add_OBV
from apps.stock_returns import get_yf_data

def Main():
    with st.sidebar.beta_expander("TA"):
        st.info(f'''
            Stock Techincal Analysis:

            * data by [yfinance](https://github.com/ranaroussi/yfinance)
            * business dates calculation by [businessdate](https://businessdate.readthedocs.io/en/latest/intro.html)
            * inspired by this [blog post](https://towardsdatascience.com/creating-a-finance-web-app-in-3-minutes-8273d56a39f8)
                and this [youtube video](https://youtu.be/OhvQN_yIgCo)
            * plots by Plotly with thanks to this [kaggle notebook](https://www.kaggle.com/mtszkw/technical-indicators-for-trading-stocks)
        ''')

    tickers = tickers_parser(st.text_input('enter stock ticker(s) [space separated]'), max_items = 1)
    with st.sidebar.beta_expander('timeframe', expanded = True):
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
        l_interval = ['1d','1m', '2m','5m','15m','30m','60m','90m','1h','5d','1wk','1mo','3mo']
        interval = st.selectbox('interval', options = l_interval)
        if interval.endswith(('m','h')):
            st.warning(f'intraday data cannot extend last 60 days')

    if tickers:
        side_config = st.sidebar.beta_expander('charts configure', expanded = False)
        with side_config:
            show_ohlc = st.checkbox('ohlc chart', value = True)
            # b_two_col = st.checkbox('two-column view', value = True)
            chart_size = st.number_input('Chart Size', value = 800, min_value = 400, max_value = 1500, step = 50)

        data_dict = get_yf_data(tickers, start_date = data_start_date, end_date = end_date, interval = interval)
        data = data_dict['prices'].copy()
        df_return = data_dict['returns'].copy()

        l_col, r_col = st.beta_columns(2)
        with l_col.beta_expander('the moving averages'):
            ma_type = st.selectbox('moving average type', options = ['', 'ema', 'sma', 'vwap'])
            periods = st.text_input('moving average periods (comma separated)', value = '22,44')
            if ma_type:
                for p in periods.split(','):
                    data = add_moving_average(data, period = int(p), type = ma_type)
        with r_col.beta_expander('MACD'):
            do_MACD = st.checkbox('Show MACD?', value = False)
            fast = st.number_input('fast', value = 12)
            slow = st.number_input('slow', value = 26)
            signal = st.number_input('signal', value = 9)
            if do_MACD:
                data = add_MACD(data, fast = fast, slow = slow, signal = signal )
        with l_col.beta_expander('oscillator'):
            data = add_AD(data) if st.checkbox('Show Advance/ Decline') else data
            data = add_OBV(data)  if st.checkbox('Show On Balance Volume') else data

        with st.beta_expander('raw data'):
            st.subheader('Price Data')
            st.write(data)
            st.subheader('Returns')
            st.write(df_return)

        l_tickers = df_return.columns.tolist()
        if len(l_tickers) != len(tickers.split(' ')):
            st.warning(f'having trouble finding the right ticker?\nCheck it out first in `DESC` :point_left:')
        single_ticker = len(l_tickers) == 1

        fig = plotly_ohlc_chart(data[data.index > pd.Timestamp(start_date)], vol_col = 'Volume')
        show_plotly(fig, height = chart_size, title = f"Price chart({interval}) for {l_tickers[0]}")

if __name__ == '__main__':
    Main()
