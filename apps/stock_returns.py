import streamlit as st
import datetime
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from businessdate import BusinessDate

from toolbox.st_utils import show_plotly, plotly_hist_draw_hline
from toolbox.yf_utils import get_stocks_data

@st.cache
def get_yf_data(tickers, start_date, end_date, interval, group_by = 'column'):
    return get_stocks_data(tickers,
        yf_download_params = {'start': start_date, 'end': end_date,
            'interval': interval, 'group_by': group_by}
        )

def compare_returns(df_returns, df_prices, chart_size = 500):
    '''
    plot multi stocks returns in ST using PX
    '''
    l_tickers = df_returns.columns.tolist()
    l_col , r_col = st.beta_columns(2)

    with l_col:
        # returns
        fig = px.line(df_returns, y = df_returns.columns.tolist(),
                    title = f'stock returns'
                    )
        show_plotly(fig, height=chart_size)

    with r_col:
        # return dist
        fig = px.histogram(df_returns, y = l_tickers,
            opacity = 0.8,
            title = f'returns distribution')
        show_plotly(fig, height = chart_size)

    # cumulative returns
    fig = px.line((df_returns+1).cumprod(), y = df_returns.columns.tolist(),
            title = f'growtht of $1 invested on {df_returns.index[0]}',
            labels = {'value': f'cumulative returns'},
            # color_discrete_sequence = ["#b58900"]
            )
    show_plotly(fig, height = chart_size)

def plot_returns(df_returns, df_prices, target_ticker,
    chart_size = 500, show_ohlc = True):
    '''
    plot target_ticker returns in ST using PX
    '''
    l_col , r_col = st.beta_columns(2)
    with l_col:
        # returns
        fig = px.line(df_returns, y = target_ticker,
                    title = f'{target_ticker} returns',
                    color_discrete_sequence = ["#b58900"],
                    )
        show_plotly(fig, height=chart_size)

        # cumulative returns
        fig = px.line((df_returns+1).cumprod(), y = target_ticker,
                title = f'growtht of $1 invested in {target_ticker} on {df_returns.index[0]}',
                labels = {target_ticker: f'{target_ticker} cumulative returns'},
                color_discrete_sequence = ["#b58900"]
                )
        show_plotly(fig, height = chart_size)

    with r_col:
        # return dist
        fig = px.histogram(df_returns, y = target_ticker,
            color_discrete_sequence = ["#b58900"],
            title = f'{target_ticker} returns distribution (with 2stdv lines)')
        two_stdv = df_returns[target_ticker].mean() + df_returns[target_ticker].std() * 2
        plotly_hist_draw_hline(fig, l_value_format = [{'value': v} for v in [two_stdv, -two_stdv]])
        show_plotly(fig, height = chart_size)

        # simple price chart
        if show_ohlc:
            fig = go.Figure(data= go.Ohlc(x = df_prices.index,
                                open= df_prices['Open'][target_ticker],
                                high= df_prices['High'][target_ticker],
                                low= df_prices['Low'][target_ticker],
                                close= df_prices['Close'][target_ticker])
                            )
        else:
            fig = px.line(df_prices['Adj Close'], y = target_ticker,
                    color_discrete_sequence = ["#b58900"])
        show_plotly(fig, height = chart_size, title=f'price of {target_ticker}')

def Main():
    with st.sidebar.beta_expander("RT"):
        st.info(f'''
            Stock Return Analysis:

            * data by [yfinance](https://github.com/ranaroussi/yfinance)
            * business dates calculation by [businessdate](https://businessdate.readthedocs.io/en/latest/intro.html)
            * inspired by this [blog post](https://www.codingfinance.com/post/2018-04-03-calc-returns-py/)
        ''')

    tickers = st.text_input('enter stock ticker(s) [space separated]')
    with st.sidebar.beta_expander('settings', expanded = True):
        today = datetime.date.today()
        end_date = st.date_input('Period End Date', value = today)
        if st.checkbox('pick start date'):
            start_date = st.date_input('Period Start Date', value = today - datetime.timedelta(days = 365))
        else:
            tenor = st.text_input('Period', value = '250b')
            start_date = (BusinessDate(end_date) - tenor).to_date()
            st.info(f'period start date: {start_date}')

        l_interval = ['1d','1m', '2m','5m','15m','30m','60m','90m','1h','5d','1wk','1mo','3mo']
        interval = st.selectbox('interval', options = l_interval)

    if tickers:
        side_config = st.sidebar.beta_expander('charts configure', expanded = False)
        with side_config:
            show_ohlc = st.checkbox('ohlc chart', value = True)
            # b_two_col = st.checkbox('two-column view', value = True)
            chart_size = st.number_input('Chart Size', value = 500, min_value = 400, max_value = 1500)


        data_dict = get_yf_data(tickers, start_date = start_date, end_date = end_date, interval = interval)
        data = data_dict['prices'].copy()
        df_return = data_dict['returns'].copy()

        with st.beta_expander('raw data'):
            st.subheader('Price Data')
            st.write(data)
            st.subheader('Returns')
            st.write(df_return)

        l_tickers = df_return.columns.tolist()
        if len(l_tickers) != len(tickers.split(' ')):
            st.warning(f'having trouble finding the right ticker?\nCheck it out first in `DESC` :point_left:')

        l_col , r_col = st.beta_columns(2)
        with l_col:
            target_ticker = st.selectbox('Analyze', options = [''] + l_tickers)

        if target_ticker:
            with r_col:
                with st.beta_expander(f'{target_ticker} descriptive stats'):
                    st.write(df_return[target_ticker].describe())
            plot_returns(df_returns = df_return, df_prices = data,
                target_ticker = target_ticker, chart_size = chart_size, show_ohlc=show_ohlc)
        else:
            compare_returns(df_returns = df_return, df_prices = data,
                chart_size = chart_size)

if __name__ == '__main__':
    Main()
