import streamlit as st
import numpy as np

from settings import GetVar
from func.newsapi import getheadlines
from func.st_utils import print_logo

@st.cache
def get_data(query_dict):
    return getheadlines(query_dict, bVerbose = True)

def print_articles(data):
    l_articles = data['articles']
    for article in l_articles:
        st.markdown(f'''
            #### [{article["title"]}]({article["url"]})\n
            *{article["source"]["name"]}*\n
            {article["description"]}
        ''')

def Main():
    print_logo()
    st.sidebar.header("Headlines")
    st.sidebar.info('''
    brought to you by [NewsAPI](https://www.alphavantage.co/documentation/), get your free API key [here](https://www.alphavantage.co/support/#api-key)

    ''')

    country = st.sidebar.text_input('Country')
    show_raw = st.sidebar.checkbox('Show raw data?')
    l_category = ['business', 'entertainment', 'general', 'health', 'science', 'sports', 'technology']
    #av_key = st.sidebar.text_input(label='Enter Your API Key')

    query_dict = {
        'q': st.sidebar.text_input(label='Search Keyword'),
        'country': country,
        'pageSize': st.sidebar.slider('pageSize', min_value = 10, max_value = 100, step = 10),
        #'language': st.sidebar.text_input('language', 'en'),
        'category': st.sidebar.selectbox('category', options = l_category)
    }
    query_dict = {k:v for k, v in query_dict.items() if v}
    if len(query_dict.items())> 0:
            data = get_data(query_dict)
            st.subheader('NewsAPI query params')
            st.json(query_dict)

            if data:
                st.subheader('News Results')
                if show_raw:
                    st.json(data)
                else:
                    print_articles(data)
            else:
                st.warning('No News Headlines Found!')


if __name__ == '__main__':
    Main()
