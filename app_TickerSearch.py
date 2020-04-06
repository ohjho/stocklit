import streamlit as st
import numpy as np

from settings import GetVar
from utils import JsonLookUp
from func.alphavantage import TickerSearch
from func.st_utils import print_logo

def SymbolSearch(keyword, alphavantage_key = None):
    alphavantage_key = alphavantage_key if alphavantage_key else GetVar("alpha_vantage_api_key")

    st.header(f"Ticker Search for '{keyword}'")
    data = TickerSearch(keyword, alphavantage_key)

    if 'bestMatches' in data.keys():
        data_syms = data['bestMatches']

        # Region Select
        lRegions = [ result['4. region'] for result in data_syms]
        lRegions = sorted(list(set(lRegions)))
        use_regions = st.sidebar.multiselect("region filter", options = lRegions, default = lRegions)

        # Display Data
        filtered_syms = [JsonLookUp(data_syms, searchKey = '4. region', searchVal = region) for region in use_regions]
        filtered_syms = [[obj] if type(obj)== dict else obj for obj in filtered_syms]
        out_syms = []
        for result in filtered_syms:
            out_syms.extend(result)

        st.subheader('Result Tickers')
        lSyms = [ result['1. symbol'] for result in out_syms]
        st.write(lSyms)

        st.subheader('Result JSON from AlphaVantage')
        st.json( out_syms)

def Main():
    print_logo()
    st.sidebar.header("Ticker Search/ Symbol Lookup")
    st.sidebar.info('''
    brought to you by [Alpha Vantage](https://www.alphavantage.co/documentation/), get your free API key [here](https://www.alphavantage.co/support/#api-key)

    challenge yourself and see if you could look for *interlisted stocks*
    (for example: [TMX](https://www.tmxmoney.com/en/research/interlisted.html),
    [HKEx](https://www.hkex.com.hk/Listing/Rules-and-Guidance/Other-Resources/Listing-of-Overseas-Companies/Company-Information-Sheets?sc_lang=en))
    or [*closed-end funds*](https://www.cefa.com/Learn/Content/CEFBasics.fs)
    (for example: [TSX](https://www.tsx.com/listings/listing-with-us/sector-and-product-profiles/closed-end-funds),
    [Nuveen](https://www.nuveen.com/closed-end-funds?term=all))
    ''')

    kw = st.sidebar.text_input(label='Search Keyword')
    av_key = st.sidebar.text_input(label='Enter Your API Key')
    #if st.sidebar.button(f'Ticker Search') and kw:
    if kw:
            SymbolSearch(kw, alphavantage_key = av_key)

if __name__ == '__main__':
    Main()
