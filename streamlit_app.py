import streamlit as st
import os, sys

from toolbox.st_utils import show_logo
from apps.yahoo_finance import Main as yf_app

def Main():
	st.set_option('deprecation.showPyplotGlobalUse', False)
	st.set_page_config(layout = 'wide')
	show_logo()
	# st.sidebar.header('OpenTerminal')
	# st.sidebar.subheader('information symmetry for all')
	with st.sidebar.beta_expander("OpenTerminal"):
		st.info(f'''
		[information symmetry](https://en.wikipedia.org/wiki/Information_asymmetry) for all
		''')

	app_dict = {
		"stock price (by yfinance)": yf_app
	}

	app_sw = st.sidebar.selectbox('select app', options = [''] + list(app_dict.keys()))
	if app_sw:
		app_func = app_dict[app_sw]
		app_func()

if __name__ == '__main__':
	Main()
