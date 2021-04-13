import streamlit as st
from pandas_finance import Equity

def Main():
    st.header("My First ST App")
    ticker = st.text_input('enter a stock ticker')
    if ticker:
        stock = Equity(ticker)
        data = stock.adj_close
        st.dataframe(data)

        st.line_chart(data)

        trading_data = stock.trading_data
        st.dataframe(trading_data)

if __name__ == '__main__':
    Main()
