import os, sys, json, requests
import streamlit as st
from stqdm import stqdm
import pandas as pd

#Paths
cwdir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, os.path.join(cwdir, "../"))
from toolbox.st_auth import run_if_auth, auth_before_run
from toolbox.st_utils import show_plotly
from toolbox.yf_utils import tickers_parser, get_stocks_obj, get_stocks_info
from toolbox.data_utils import JsonReader, JsonLookUp
from toolbox.hkex_utils import get_hangseng_constituent_df

STOCK_UNIVERSE = JsonReader(os.path.join(cwdir,'../data/index_definition.json'))

def get_index_members(index_name, index_dicts = STOCK_UNIVERSE, limit = None):
    '''
    return a list of yf tickers for the given index
    references:
        https://medium.com/wealthy-bytes/5-lines-of-python-to-automate-getting-the-s-p-500-95a632e5e567
        https://medium.com/financial-data-analysis/step-1-web-scraping-hong-kong-hsi-stock-price-7d8606c07c57
        https://tcoil.info/build-simple-stock-trading-bot-advisor-in-python/
    '''
    if index_name not in [d['index'] for d in index_dicts]:
        return None
    else:
        #TODO: check idict['url'] if is .xls or .csv do something else
        idict = JsonLookUp(index_dicts, searchKey = 'index', searchVal = index_name)
        tables = pd.read_html(
                    requests.get(idict['url'], headers={'User-agent': 'Mozilla/5.0'}).text,
                    header= 0)
        df = tables[idict['df']]

        # Special Handling
        if index_name.startswith('^DVD'):
            df['Symbol'] = df['Company'].apply( lambda x: x.split()[-1].split(":")[-1].replace(")",''))
        elif index_name in ['^NDX']:
            df['Symbol'] = df['Ticker']
        elif index_name == '^TX60':
            df = df.dropna(subset = ['Symbol'])
            df['Symbol'] = df['Symbol'].apply(lambda x: str(x).replace('.', '-')+'.TO')
        elif index_name == '^HCS':
            df = get_hangseng_constituent_df()
        elif index_name in ['^HSI','^HCM','^HCL', '^HSTECH', '^H35']:
            df['Symbol'] = [str(s).zfill(4)+'.HK' for s in df['Code'].tolist()]
        # TODO: apply limit
        return df['Symbol'].tolist()

@st.cache
def get_etf_holdings(etf_ticker, parse = False):
    '''
    prototype function, doesn't work yet
    '''
    # TODO: need chromium webdriver
    # see: https://medium.com/hackernoon/python-notebook-research-to-replicate-etf-using-free-data-ca9f88eb7349
    # or scrap zacks: https://stackoverflow.com/questions/64908086/using-python-to-identify-etf-holdings
    ref_url = f'https://www.barchart.com/etfs-funds/quotes/{etf_ticker}/constituents?page=all'
    # tables = pd.read_html(ref_url, header = {'User-Agent': 'Mozilla/5.0'})
    tables = pd.read_html(requests.get(ref_url,
                                   headers={'User-agent': 'Mozilla/5.0'}).text,
                      attrs={"class":"constituents"} if parse else None)
    print(len(tables))
    df = tables[2]
    return df

def showIndices(l_indices = STOCK_UNIVERSE, st_asset = st, as_df = False):
    with st_asset.expander('available indices'):
        if as_df:
            df = pd.DataFrame(l_indices).set_index('index')
            st.write(df)
        else:
            for i in l_indices:
                st.write(f'`{i["index"]}`: [{i["name"]}]({i["url"]})')

@st.cache(suppress_st_warning=True)
def get_members_info(asset, tqdm_func = stqdm):
    '''
    return a list of json object containing info for each member within the asset
    Args:
        asset: Index name (must be in STOCK_UNIVERSE) or list of tickers
    '''
    l_tickers = None
    if type(asset) == list:
        l_tickers = asset
    elif type(asset) == str:
        l_tickers = get_index_members(asset)

    if l_tickers:
        results = get_stocks_info(" ".join(l_tickers), tqdm_func = tqdm_func)
        return results
    else:
        return None

@st.cache
def get_members_info_df(asset, l_keys = ['symbol', 'longName']):
    info_json = get_members_info(asset = asset)
    df = pd.DataFrame(info_json)
    return df[l_keys]

def get_index_tickers(st_asset = st.sidebar):
    with st_asset:
        l_indices = [d['index'] for d in STOCK_UNIVERSE]
        idx = st.selectbox('Index', options = [''] + l_indices)

        if idx:
            l_members = get_index_members(index_name = idx)
            ref_security = JsonLookUp(STOCK_UNIVERSE,
                            searchKey = 'index', searchVal = idx, resultKey = 'reference_security')
            st.info(f'''
                Found {len(l_members)} index members and reference security: {ref_security}
            ''')
            if st.checkbox('Load members to tickers field', value = False):
                return ' '.join(l_members)
            else:
                return ''
        else:
            return ''

@auth_before_run
def Main():
    with st.sidebar.expander("MBRS"):
        st.info(f'''
            Getting Indices members and ETFs holdings (coming soon)

            * data by [yfinance](https://github.com/ranaroussi/yfinance)
        ''')

    showIndices(st_asset = st.sidebar)
    default_tickers = get_index_tickers(
                        st_asset = st.sidebar.expander('Load an Index', expanded = True)
                        )

    with st.sidebar.expander('settings', expanded = False):
        df_height = int(st.number_input("members' df height", value = 500, min_value = 200))

    tickers = tickers_parser(
                st.text_input("index members' tickers [space separated]",
                    value = default_tickers)
                )

    if tickers:
        with st.expander('display keys'):
            l_col, r_col = st.columns(2)
            with l_col:
                l_keys_des = st.multiselect('descriptive',
                                options = ['longName', 'previousClose','sector', 'fullTimeEmployees', 'country', 'industry', 'currency', 'exchangeTimezoneName'],
                                default = ['longName'])
                l_keys_vol = st.multiselect('volume',
                                options = ['averageVolume10days', 'circulatingSupply', 'sharesOutstanding', 'sharesShort','sharesPercentSharesOut', 'floatShares', 'shortRatio', 'heldPercentInsiders', 'impliedSharesOutstanding']
                            )
            with r_col:
                l_keys_dvd = st.multiselect('dividend related',
                                options = ['dividendRate', 'exDividendDate', 'dividendYield', 'lastDividendDate', 'exDividendDate', 'lastDividendValue']
                            )
                l_keys_fun = st.multiselect('fundamental',
                                options = ['marketCap','trailingPE','priceToSalesTrailing12Month','forwardPE', 'profileMargins', 'forwardEps','bookValue', 'priceToBook', 'payoutRatio']
                            )
        l_keys = l_keys_des  + l_keys_vol + l_keys_dvd + l_keys_fun
        if len(l_keys) < 1:
            st.warning(f'no key selected.')
            return None

        # st.subheader(f'Members of `{idx}`')
        st.subheader(f'Index Members stats')
        data = get_members_info_df(asset = tickers.split(), l_keys=['symbol'] + l_keys)
        st.dataframe(data, height = df_height)

        # TODO: ticker selector to return a space-separated string for use in other apps

if __name__ == '__main__':
    Main()
