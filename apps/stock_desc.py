import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

from toolbox.yf_utils import valid_stock, tickers_parser
from toolbox.plotly_utils import plotly_ohlc_chart

def Main():
    ticker = tickers_parser(st.text_input('enter a stock ticker'), max_items = 1)
    with st.sidebar.beta_expander("DESC"):
        st.info(f'''
            Historical Stock Price by [yfinance](https://github.com/ranaroussi/yfinance)
            * [yahoo finance ticker lookup](https://finance.yahoo.com/lookup)
            * [blog post](https://aroussi.com/post/python-yahoo-finance)
        ''')

    if ticker:
        # ticker = ticker.split('.')[0].zfill(4) + '.HK' if '.HK' in ticker.upper() else ticker
        side_config = st.sidebar.beta_expander('configure', expanded = True)
        with side_config:
            show_ohlc = st.checkbox('ohlc chart', value = True)
            show_volume = st.checkbox('show volume', value = True)
            b_two_col = st.checkbox('two-column view', value = True)
            chart_size = st.number_input('Chart Size', value = 500, min_value = 400, max_value = 1500)

        if b_two_col:
            col1, col2 = st.beta_columns((2,1))
        else:
            col1 = col2 = st.beta_container()

        stock = yf.Ticker(ticker)
        if not valid_stock(stock):
            st.error(f'Cannot find `{ticker}`')
            return None

        #TODO:
        # 1. show key keys info
        # 2. add twitter & NewsAPI
        # 4. Download price DF

        with col1:
            stock_info = stock.info
            with st.beta_expander(f'Stock Info for {stock_info["longName"]}'):
                str_desc = f'[:link:]({stock_info["website"]}) ' \
                            if 'website' in stock_info.keys() else ''
                str_desc += stock_info["longBusinessSummary"] \
                            if "longBusinessSummary" in stock_info.keys() else ":warning: Business Summary not available"
                st.write(f'{str_desc}')
                if st.checkbox('show info JSON'):
                    st.json(stock_info)

            df_all = stock.history(period = "max")
            with side_config:
                number_td = st.number_input("Number of trading days", value = min(250, len(df_all)),
                                min_value = 5, max_value = len(df_all))
            df_all = df_all.tail(number_td)

            with st.beta_expander('Price Chart', expanded = True):
                if show_ohlc:
                    fig = plotly_ohlc_chart(df = df_all, vol_col = 'Volume' if show_volume else None)
                else:
                    fig = px.line(df_all, y = 'Close',
                        color_discrete_sequence = ["#b58900"])
                        # color_discrete_sequence = ['yellow'])
                fig.update_layout(height = chart_size,
                    title = f'{ticker.upper()} last {number_td} trading days- ohlc chart',
                    template = 'plotly_dark')
                st.plotly_chart(fig, use_container_width = True, height = chart_size)

            with st.beta_expander('Historical Prices'):
                st.dataframe(df_all)

            # High, Low, Open, Close, Volume, Adj Close
            # trading_data = stock.trading_data
            # st.dataframe(trading_data)

            # show actions (dividends, splits)
            with st.beta_expander("Corporate Actions"):
                st.dataframe(stock.actions)

            # show dividends
            with st.beta_expander("Dividends"):
                st.dataframe(stock.dividends)

            # show splits
            with st.beta_expander("Splits"):
                st.dataframe(stock.splits)

        with col2:
            # show financials
            with st.beta_expander("Financials"):
                st.write(stock.financials)
                # msft.quarterly_financials

            # show major holders
            with st.beta_expander("Major Holders"):
                st.dataframe(stock.major_holders)

            # show institutional holders
            with st.beta_expander("Institutional Holders"):
                st.dataframe(stock.institutional_holders)

            # show balance sheet
            with st.beta_expander("Balance Sheet"):
                st.write(stock.balance_sheet)
                # msft.quarterly_balance_sheet

            # show cashflow
            with st.beta_expander("Cashflow"):
                st.write(stock.cashflow)
                # msft.quarterly_cashflow

            # show earnings
            with st.beta_expander("Earnings"):
                st.write(stock.earnings)
                # msft.quarterly_earnings

            # show sustainability
            with st.beta_expander("Sustainability"):
                st.dataframe(stock.sustainability)

            # show analysts recommendations
            with st.beta_expander("Analysts Recommendations"):
                # TODO:
                # 1. set "Firm" as index
                # 2. turn datetime index as a col
                st.dataframe(stock.recommendations)

            # show next event (earnings, etc)
            with st.beta_expander("Calendar"):
                if isinstance(stock.calendar, pd.DataFrame):
                    st.dataframe(stock.calendar.T)

            # show ISIN code - *experimental*
            # with st.beta_expander("ISIN (experimental)"):
            # ISIN = International Securities Identification Number
                # print(stock.isin)
                # print(type(stock.isin))
                # st.write(f"{stock.isin}")


if __name__ == '__main__':
    Main()