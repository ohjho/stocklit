import os, sys, json, datetime
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from businessdate import BusinessDate
from stqdm import stqdm

#Paths
cwdir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, os.path.join(cwdir, "../"))
from toolbox.st_utils import get_json_edit
from toolbox.yf_utils import tickers_parser
from toolbox.data_utils import JsonLookUp
from apps.stock_members import get_index_tickers
from strategies.utils import get_ticker_data_with_technicals, \
                            backtest, analysis_bt
from strategies.vol_breakout import detect_vol_breakout, minervini_vcp, detect_VCP, alpha_over_beta
# from strategies.macd_divergence import add_strategy_buy

def get_strategy_params(st_asset = st.sidebar):
    with st_asset:
        if st.checkbox('try alpha_over_beta'):
            return {}

        # Vol Breakout Params
        params = {
            'atr_threshold': st.number_input('vol breakout threshold', value = 1.0),
            'atr_period': st.number_input('ATR period', value = 10),
            'ignore_volume': st.checkbox("ignore volume", value = False),
            'ignore_gap': st.checkbox("ignore gap", value = True),
        }

        # VCP Params
        d_vcp_params = {'ma_cascade' : [100,50,25], 'lvpb_period' : 22,
                'normalize_ATR' : False, 'debug_mode': True}
        vcp_params = get_json_edit(d_vcp_params, str_msg = 'Strategy Params',
                        text_area_height = 150)\
                        if st.checkbox('filter for Volatility Contraction') else {}
        params['vcp_params'] = vcp_params

    # for k,v in params:
    #     if not v:
    #         return None
    return params

def get_backtest_params(st_asset = st.sidebar):
    with st_asset:
        params_default = {
            'signal_col': 'vol_breakout',
            'fair_value_col': 'entry_price',
            'stop_atr_ratio': 1, 'rr_ratio': 10,
            'breakeven_stop_r': 1.5, 'trailing_stop_atr': 3,
            'profit_retracement': 0.3
        }
        params_str = st.text_area('Backtest Parameters',
                        value = json.dumps(params_default),
                        height = 200
                        )
    return json.loads(params_str)

@st.cache(suppress_st_warning=True) # because of the use of stqdm
def backtest_collection(data_collection, bt_params, strategy_params,
                    min_data_points = 200, strategy_func = detect_vol_breakout,
                    debug = True):
    notification = []
    trades = []
    for d in stqdm(data_collection, desc = 'Backtesting Strategy'):
        if len(d['df'])< min_data_points:
            print(f'{d["ticker"]} has less than {min_data_points} data points')
            continue
        # Run Strategy
        d['df'] = strategy_func(d['df'] if debug else d['df'].copy(), **strategy_params)

        if any(d['df'][bt_params['signal_col']][-5:]):
            notification.append(d['ticker'])

        # Back Test
        d['trades'] = backtest(d['df'], ticker = d['ticker'], **bt_params)
        if len(d['trades'])> 0:
            trades += d['trades']

    return trades, notification

def bt_results_eda(result_object):
    results = result_object
    l_col, r_col = st.beta_columns(2)
    l_col.write({k:v for k,v in results.items() if k != 'df'})

    df_p = results['df']['market_classification_3m'].value_counts()
    fig_mkt_class = px.pie(df_p, values = 'market_classification_3m',
                        names = df_p.index.tolist(),
                        color = df_p.index.tolist(),
                        title = "Market Classification at Entry"
                        )
    r_col.plotly_chart(fig_mkt_class, use_container_width = True)

    with l_col:
        mkt_class = st.selectbox('Filter Trades by Market Classification',
                        options = [''] + df_p.index.tolist()
                        )


    df_p = results['df']
    df_p = df_p[df_p['market_classification_3m']==mkt_class] if mkt_class else df_p
    fig_r_dist = px.histogram(df_p, x = 'r_multiplier',
                    color = 'ticker', barmode= 'stack', nbins= 100,
                    title = 'All Trades R-ratio Distribution')
    fig_holding_period = px.histogram(df_p, x = 'num_days',
                    color = 'profitable', barmode= 'stack', nbins= 50,
                    color_discrete_map = {True: 'green', False: 'red'},
                    title = 'All Trades Holding Period Distribution')
    l_col, r_col = st.beta_columns(2)
    st.plotly_chart(fig_r_dist, use_container_width = True)
    st.plotly_chart(fig_holding_period, use_container_width = True)

    # winning vs losing trades

    df_p = results['df']
    fig_mae = px.histogram(df_p, x = 'MAE',
                    color = 'profitable', barmode= 'stack', nbins= 50,
                    color_discrete_map = {True: 'green', False: 'red'},
                    title = 'Maximum Adverse Excurison (as R-Multiples)')
    st.plotly_chart(fig_mae, use_container_width = True)

    fig_equity_curve = px.line(x = df_p['date'],
                            y = df_p["r_multiplier"].cumsum(),
                            # markers = True,
                            labels = {'x': "date", 'y': "r-r_multiplier (cumulative)"},
                            title = f"Equity Curve from {df_p['date'][0]} to {df_p['date'].tolist()[-1]}"
                        )
    st.plotly_chart(fig_equity_curve, use_container_width = True)

def Main():
    with st.sidebar.beta_expander('Backtest: Volatility Breakout'):
        st.info(f'''
            Volatility Breakout
            ''')

    with st.sidebar.beta_expander('timeframe', expanded = True):
        today = datetime.date.today()
        end_date = st.date_input('Period End Date', value = today)
        if st.checkbox('pick start date'):
            start_date = st.date_input('Period Start Date', value = today - datetime.timedelta(days = 365))
        else:
            tenor = st.text_input('Period', value = '5y')
            start_date = (BusinessDate(end_date) - tenor).to_date()
            st.info(f'period start date: {start_date}')
        scan_mode = st.checkbox('Scanner Mode', value = True)
        bar_interval = st.selectbox('Interval', options = ['1d', '1wk'])

    default_tickers = get_index_tickers(
                        st.sidebar.beta_expander("Load Tickers from an Index", expanded = True)
                        )
    tickers = tickers_parser(st.text_input('enter stock ticker(s)', value = default_tickers),
                return_list = True, max_items = None)

    if tickers:
        cols = st.beta_columns(2)
        strategy_params = get_strategy_params(cols[0].beta_expander('Strategy Params'))
        bt_params = get_backtest_params(cols[1].beta_expander('Backtest Params'))

        data_collection = [
            {'ticker': t, 'df': get_ticker_data_with_technicals(ticker = t,
                    interval = bar_interval, trade_date = end_date,
                    data_buffer_tenor = '3y',
                    tenor = tenor)
            }
            for t in stqdm(tickers, desc = 'Downloading Data')
        ]

        trades, notification = backtest_collection(data_collection, bt_params,
                                    strategy_params, min_data_points = 200 if bar_interval == '1d' else 100,
                                    strategy_func= minervini_vcp if strategy_params else alpha_over_beta)

        if len(notification)> 0:
            with st.beta_expander(f'tickers showing Vol Breakout in last 5 bars',
                        expanded = True):
                st.write(notification)

        if scan_mode and strategy_params['vcp_params']:
            # vcp_notify = [d['ticker'] for d in stqdm(data_collection, desc='checking for VCP')
            #     if detect_VCP(d['df'], ATR_period =strategy_params['atr_period'],**strategy_params['vcp_params'])
            #     ]
            objects_with_vcp = [d for d in data_collection if 'VCP_setup' in d['df'].columns]
            vcp_notify = [d['ticker'] for d in objects_with_vcp 
                    if any(d['df']['VCP_setup'][-5:])
                    ]
            if vcp_notify:
                with st.beta_expander(f'tickers in VCP', expanded = True):
                    st.write(vcp_notify)

        with st.beta_expander('View Data'):
            t = st.selectbox('ticker', options = [d['ticker'] for d in data_collection])
            st.write(JsonLookUp(data_collection, searchKey = 'ticker', searchVal = t, resultKey = 'df'))

        # show results
        st.subheader('Backtest results')
        if len(trades)> 0:
            results = analysis_bt([t for t in trades if 'r_multiplier' in t.keys()],
                        n_trading_days = (end_date - start_date).days)
            with st.beta_expander('view trades'):
                st.write(results['df'])

                open_trades = [t for t in trades if 'r_multiplier' not in t.keys()]
                if open_trades:
                    st.write('opened trades')
                    df_open = pd.DataFrame(open_trades)
                    df_open['r_multiplier'] = (df_open['stop'] - df_open['entry'])/df_open['risk']
                    st.write(df_open)

            with st.beta_expander('visualize backtest results', expanded = True):
                bt_results_eda(results)
        else:
            st.warning(f'Found nothing during backtest')

if __name__ == '__main__':
    Main()
