import os, sys, json, datetime
import streamlit as st
import pandas as pd
import plotly.express as px
from businessdate import BusinessDate

#Paths
cwdir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, os.path.join(cwdir, "../"))
from toolbox.st_utils import show_plotly, plotly_hist_draw_hline
from toolbox.yf_utils import tickers_parser, get_stocks_data
from toolbox.plotly_utils import plotly_ohlc_chart, get_moving_average_col, \
                            add_Scatter, add_Scatter_Event
from toolbox.ta_utils import add_moving_average, add_MACD, add_AD, add_OBV, add_RSI, \
                            add_ADX, add_Impulse, add_ATR, add_avg_penetration, \
                            market_classification, \
                            detect_kangaroo_tails, detect_macd_divergence
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

    tickers = tickers_parser(st.text_input('enter stock ticker(s)'), max_items = 1)
    with st.sidebar.beta_expander('timeframe', expanded = True):
        today = datetime.date.today()
        end_date = st.date_input('Period End Date', value = today)
        if st.checkbox('pick start date'):
            start_date = st.date_input('Period Start Date', value = today - datetime.timedelta(days = 365))
        else:
            tenor = st.text_input('Period', value = '6m')
            start_date = (BusinessDate(end_date) - tenor).to_date()
            st.info(f'period start date: {start_date}')

        # TODO: allow manual handling of data_start_date
        l_interval = ['1d','1wk','1m', '2m','5m','15m','30m','60m','90m','1h','5d','1mo','3mo']
        interval = st.selectbox('interval', options = l_interval)
        is_intraday = interval.endswith(('m','h'))
        data_start_date = start_date if is_intraday else \
                        (BusinessDate(start_date) - "1y").to_date()
        if is_intraday:
            st.warning(f'''
                intraday data cannot extend last 60 days\n
                also, some features below might not work properly
                ''')

    if tickers:
        side_config = st.sidebar.beta_expander('charts configure', expanded = False)
        with side_config:
            show_ohlc = st.checkbox('ohlc chart', value = True)
            # b_two_col = st.checkbox('two-column view', value = True)
            chart_size = st.number_input('Chart Size', value = 800, min_value = 400, max_value = 1500, step = 50)

        data_dict = get_yf_data(tickers, start_date = data_start_date, end_date = end_date, interval = interval)
        data = data_dict['prices'].copy()
        df_return = data_dict['returns'].copy()

        with st.beta_expander('Indicators'):
            l_col, m_col , r_col = st.beta_columns(3)
            with l_col:
                st.write('#### the moving averages')
                ma_type = st.selectbox('moving average type', options = ['', 'ema', 'sma', 'vwap'])
                periods = st.text_input('moving average periods (comma separated)', value = '22,11')
                if ma_type:
                    for p in periods.split(','):
                        data = add_moving_average(data, period = int(p), type = ma_type)
                st.write('#### volume-based indicators')
                # do_volume_profile = st.checkbox('Volume Profile')
                data = add_AD(data) if st.checkbox('Show Advance/ Decline') else data
                data = add_OBV(data)  if st.checkbox('Show On Balance Volume') else data
            with m_col:
                st.write('#### MACD')
                do_MACD = st.checkbox('Show MACD?', value = False)
                fast = st.number_input('fast', value = 12)
                slow = st.number_input('slow', value = 26)
                signal = st.number_input('signal', value = 9)
                if do_MACD:
                    data = add_MACD(data, fast = fast, slow = slow, signal = signal )
            with r_col:
                st.write('#### oscillator')
                do_RSI = st.checkbox('RSI')
                data = add_RSI(data, n = st.number_input('RSI period', value = 13)) if do_RSI else data
                tup_RSI_hilo = st.text_input('RSI chart high and low line (comma separated):', value = '70,30').split(',') \
                                if do_RSI else None
                tup_RSI_hilo = [int(i) for i in tup_RSI_hilo] if tup_RSI_hilo else None
                if do_RSI:
                    data_over_hilo_pct = sum(
                        ((data['RSI']> tup_RSI_hilo[0]) | (data['RSI']< tup_RSI_hilo[1])) & (data.index > pd.Timestamp(start_date))
                        ) / len(data[data.index > pd.Timestamp(start_date)])
                    st.info(f"""
                    {round(data_over_hilo_pct * 100, 2)}% within hilo\n
                    5% of peaks and valley should be within hilo
                    """)

                st.write('#### True Range Related')
                atr_period = st.number_input('Average True Range Period', value = 13)
                atr_ema = st.checkbox('use EMA for ATR', value = True)
                if ma_type:
                    st.write('##### ATR Channels')
                    atr_ma_name = st.selectbox('select moving average for ATR channel',
                                    options = [''] + get_moving_average_col(data.columns))
                    atr_channels = st.text_input('Channel Lines (comma separated)', value = "1,2,3") \
                                    if atr_ma_name else None
                    fill_channels = st.checkbox('Fill Channels with color', value = False) \
                                    if atr_ma_name else None
                else:
                    atr_ma_name = None

                data = add_ATR(data, period = atr_period, use_ema = atr_ema,
                            channel_dict = {atr_ma_name: [float(c) for c in atr_channels.split(',')]} \
                                if atr_ma_name else None
                            )
                st.write(f'##### Directional System')
                do_ADX = st.checkbox('Show ADX')
                data = add_ADX(data, period = st.number_input("ADX period", value = 13)) \
                        if do_ADX else data

        with st.beta_expander('advanced settings'):
            l_col, m_col , r_col = st.beta_columns(3)
            with l_col:
                st.write('#### Market Type Classification')
                mkt_class_period = st.number_input('peroid (match your trading time domain)', value = 66)
                mkt_class = market_classification(data, period = mkt_class_period,
                                debug = False) if mkt_class_period else None
                if mkt_class:
                    st.write(f'market is `{mkt_class}` for the last **{mkt_class_period} bars**')

                if do_MACD and ma_type:
                    st.write("#### Elder's Impulse System")
                    impulse_ema = st.selectbox('select moving average for impulse',
                                    options = [''] + get_moving_average_col(data.columns))
                    data = add_Impulse(data, ema_name = impulse_ema) if impulse_ema else data

            avg_pen_data = None
            with m_col:
                if ma_type:
                    st.write("#### Average Penetration for Entry/ SafeZone")
                    fair_col = st.selectbox('compute average penetration below',
                                    options = [''] + get_moving_average_col(data.columns))
                    avg_pen_data = add_avg_penetration(df = data, fair_col = fair_col,
                                        num_of_bars = st.number_input('period (e.g. 4-6 weeks)', value = 30), # 4-6 weeks
                                        use_ema = st.checkbox('use EMA for penetration', value = False),
                                        ignore_zero = st.checkbox('ignore days without penetration', value = True),
                                        coef = st.number_input(
                                            'SafeZone Coefficient (stops should be set at least 1x Average Penetration)',
                                            value = 1.0, step = 0.1),
                                        get_df = True, debug = True
                                    ) if fair_col else None
            with r_col:
                if do_MACD:
                    st.write('#### MACD Bullish Divergence')
                    if st.checkbox('Show Divergence'):
                        data = detect_macd_divergence(data,
                                period = st.number_input('within number of bars (should be around 3 months)', value = 66),
                                threshold = st.number_input('current low threshold (% of previous major low)', value = 0.95),
                                debug = True
                                )
                st.write(f'#### Detect Kangaroo Tails')
                tail_type = st.selectbox('Tail Type',
                                options = ['', 0, 1, -1])
                data = detect_kangaroo_tails(data,
                        atr_threshold = st.number_input('ATR Threshold', value = 2.0),
                        period = st.number_input('period', value = 22), tail_type = tail_type) \
                        if tail_type else data

        with st.sidebar.beta_expander(tickers, expanded = True):
            if mkt_class:
                st.write(f'market is `{mkt_class}` for the last **{mkt_class_period} bars**')

            if tickers.endswith(('.HK', '.TO')):
                st.write(f'''
                * [news link](https://www.reuters.com/companies/{tickers}/news)
                ''')

        with st.beta_expander(f'raw data (last updated: {data.index[-1].strftime("%c")})'):
            # st.subheader('Price Data')
            st.write(data)
            # st.subheader('Returns')
            # st.write(df_return)

        if isinstance(avg_pen_data, pd.DataFrame):
            with st.beta_expander('Buy Entry (SafeZone)'):
                avg_pen_dict = {
                    'average penetration': avg_pen_data['avg_lp'][-1],
                    'ATR': avg_pen_data['ATR'][-1],
                    'penetration stdv': avg_pen_data['std_lp'][-1],
                    'number of penetrations within period': avg_pen_data['count_lp'][-1],
                    'last': avg_pen_data['Close'][-1],
                    'expected ema T+1': avg_pen_data[fair_col][-1] + (avg_pen_data[fair_col][-1] - avg_pen_data[fair_col][-2])
                    }
                avg_pen_dict = {k:round(v,2) for k,v in avg_pen_dict.items()}
                avg_pen_dict['buy target T+1'] = avg_pen_dict['expected ema T+1'] - avg_pen_dict['average penetration']
                st.write(avg_pen_dict)
                plot_avg_pen = st.checkbox('plot buy SafeZone and show average penetration df')
                plot_target_buy = False # st.checkbox('plot target buy T+1')
                # if plot_avg_pen:
                #     st.write(avg_pen_data)

        l_tickers = df_return.columns.tolist()
        if len(l_tickers) != len(tickers.split(' ')):
            st.warning(f'having trouble finding the right ticker?\nCheck it out first in `DESC` :point_left:')
        single_ticker = len(l_tickers) == 1

        #TODO: fix tz issue for interval < 1d
        # see: https://stackoverflow.com/questions/16628819/convert-pandas-timezone-aware-datetimeindex-to-naive-timestamp-but-in-certain-t
        fig = plotly_ohlc_chart(
                df = data if is_intraday else data[data.index > pd.Timestamp(start_date)],
                vol_col = 'Volume',
                tup_rsi_hilo = tup_RSI_hilo,
                b_fill_channel = fill_channels if atr_ma_name else None
                ) #, show_volume_profile = do_volume_profile)
        # SafeZone
        if isinstance(avg_pen_data, pd.DataFrame):
            fig = add_Scatter(fig, df = avg_pen_data[avg_pen_data.index > pd.Timestamp(start_date)], target_col = 'buy_safezone') \
                if plot_avg_pen else fig
            if plot_target_buy:
                fig.add_hline(y = avg_pen_dict['buy target T+1'] , line_dash = 'dot', row =1, col = 1)
        # Events
        for d in ['MACD_Divergence', 'kangaroo_tails']:
            if d in data.columns:
                fig = add_Scatter_Event(fig, data[data.index > pd.Timestamp(start_date)],
                        target_col = d,
                        anchor_col = 'Low', textposition = 'bottom center', fontsize = 8,
                        marker_symbol = 'triangle-up', event_label = d[0])

        show_plotly(fig, height = chart_size, title = f"Price chart({interval}) for {l_tickers[0]}")

if __name__ == '__main__':
    Main()
