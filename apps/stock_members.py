import os, sys, json, requests
import streamlit as st
from stqdm import stqdm
import pandas as pd

#Paths
cwdir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, os.path.join(cwdir, "../"))
from toolbox.st_utils import show_plotly
from toolbox.yf_utils import tickers_parser, get_stocks_obj, get_stocks_info
from toolbox.data_utils import JsonLookUp

STOCK_UNIVERSE = [
    {'name': 'S&P 500', 'index': '^GSPC', 'reference_security': 'SPY',
        'url': 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
        'df': 0
    },
    {'name': 'Hang Seng Index', 'index': '^HSI', 'reference_security': '2800.HK',
        'url': 'http://www.etnet.com.hk/www/eng/stocks/indexes_detail.php?subtype=HSI',
        'df': 2
    },
    {'name': 'Hang Seng Mid-cap Index', 'index': '^HCM', 'reference_security': '2800.HK',
        'url': 'http://www.etnet.com.hk/www/eng/stocks/indexes_detail.php?subtype=HCM',
        'df': 2
    },
    {'name': 'US Dividend Kings', 'index': '^DVDKING', 'reference_security': 'SPY',
        'url': 'https://www.fool.com/investing/stock-market/types-of-stocks/dividend-stocks/dividend-kings/',
        'df': 0
    }
]

@st.cache
def get_index_members(index_name, index_dicts = STOCK_UNIVERSE, limit = None):
    '''
    return a list of yf tickers for the given index
    '''
    if index_name == '^GSPC':
        # ref: https://medium.com/wealthy-bytes/5-lines-of-python-to-automate-getting-the-s-p-500-95a632e5e567
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = tables[0]
        return df['Symbol'].tolist()
    elif index_name == '^HSI':
        # see https://tcoil.info/build-simple-stock-trading-bot-advisor-in-python/
        # tables = pd.read_html('https://finance.yahoo.com/quote/%5EHSI/components/')
        # df = tables[0]
        # return df['Symbol'].tolist()

        # see https://medium.com/financial-data-analysis/step-1-web-scraping-hong-kong-hsi-stock-price-7d8606c07c57
        tables = pd.read_html(requests.get('http://www.etnet.com.hk/www/eng/stocks/indexes_detail.php?subtype=HSI',
                                       headers={'User-agent': 'Mozilla/5.0'}).text,
                                       header= 0)
        df = tables[2]
        return [str(s).zfill(4)+'.HK' for s in df['Code'].tolist()]
    elif index_name == '^HCM':
        tables = pd.read_html(requests.get('http://www.etnet.com.hk/www/eng/stocks/indexes_detail.php?subtype=HCM',
                                       headers={'User-agent': 'Mozilla/5.0'}).text,
                                       header= 0)
        df = tables[2]
        return [str(s).zfill(4)+'.HK' for s in df['Code'].tolist()]
    else:
        # TODO: future of this function
        if index_name not in [d['index'] for d in index_dicts]:
            return None
        else:
            idict = JsonLookUp(index_dicts, searchKey = 'index', searchVal = index_name)
            tables = pd.read_html(
                        requests.get(idict['url'], headers={'User-agent': 'Mozilla/5.0'}).text,
                        header= 0)
            df = tables[idict['df']]

            # Special Handling
            if index_name.startswith('^DVD'):
                df['Symbol'] = df['Company'].apply( lambda x: x.split()[-1].split(":")[-1].replace(")",''))
                return df['Symbol'].tolist()

            # TODO: apply limit

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
    with st_asset.beta_expander('available indices'):
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

def GetUserInput(st_asset = st.sidebar):
    pass

def Main():
    with st.sidebar.beta_expander("MBRS"):
        st.info(f'''
            Getting Indices members and ETFs holdings (coming soon)

            * data by [yfinance](https://github.com/ranaroussi/yfinance)
        ''')

    showIndices(st_asset = st.sidebar)

    with st.sidebar.beta_expander('Load an Index', expanded = True):
        l_indices = [d['index'] for d in STOCK_UNIVERSE]
        idx = st.selectbox('Index', options = [''] + l_indices)

        if idx:
            l_members = get_index_members(index_name = idx)
            ref_security = JsonLookUp(STOCK_UNIVERSE,
                            searchKey = 'index', searchVal = idx, resultKey = 'reference_security')
            st.info(f'''
                Found {len(l_members)} index members and reference security: {ref_security}
            ''')

    tickers = tickers_parser(
                st.text_input("index members' tickers [space separated]",
                    value = " ".join(l_members) if idx else '')
                # value = ref_security + " " + " ".join(l_members) if idx else '')
                )
    with st.sidebar.beta_expander('settings', expanded = False):
        df_height = st.number_input("members' df height", value = 500, min_value = 200)

    if tickers:
        with st.beta_expander('display keys'):
            l_col, r_col = st.beta_columns(2)
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

        st.subheader(f'Members of `{idx}`')
        data = get_members_info_df(asset = tickers.split(), l_keys=['symbol'] + l_keys)
        st.dataframe(data, height = df_height)
    # if tickers:
    #     side_config = st.sidebar.beta_expander('charts configure', expanded = False)
    #     with side_config:
    #         show_ohlc = st.checkbox('ohlc chart', value = True)
    #         # b_two_col = st.checkbox('two-column view', value = True)
    #         chart_size = st.number_input('Chart Size', value = 500, min_value = 400, max_value = 1500)
    #
    #
    #     data_dict = get_yf_data(tickers, start_date = start_date, end_date = end_date, interval = interval)
    #     df_returns = data_dict['returns'].copy()
    #
    #     with st.beta_expander('view returns data'):
    #         st.subheader('Returns')
    #         st.write(df_returns)
    #
    #     l_tickers = df_returns.columns.tolist()
    #     if len(l_tickers) != len(tickers.split(' ')):
    #         st.warning(f'having trouble finding the right ticker?\nCheck it out first in `DESC` :point_left:')
    #
    #     l_col, r_col = st.beta_columns(2)
    #     with l_col:
    #         benchmark_ticker = st.selectbox('benchmark security', options = tickers.split())
    #         beta_json = get_betas(df_returns, benchmark_col = benchmark_ticker.upper())
    #         # TODO: add dividend yield?
    #         beta_df = pd.DataFrame.from_dict(beta_json)
    #     with r_col:
    #         plot_var_options = [col for col in beta_df.columns if col not in ['ticker']]
    #         y_var = st.selectbox('y-axis variable', options = plot_var_options)
    #         x_var = st.selectbox('x-axis variable', options = plot_var_options)
    #
    #     with st.beta_expander('Betas Calcuation'):
    #         st.write(beta_df)
    #
    #     fig = px.scatter(beta_df, x = x_var, y = y_var, color = 'ticker')
    #     fig.update_layout(showlegend = False)
    #     show_plotly(fig)
    #
    #     #TODO: individual stock shows beta vs return over time

if __name__ == '__main__':
    Main()
