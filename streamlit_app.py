import streamlit as st
import os, sys

import appdirs as ad
ad.user_cache_dir = lambda *args: "/tmp"

from toolbox.st_utils import show_logo
from apps.login import Main as login
from apps.stock_desc import Main as Desc
from apps.stock_returns import Main as RT
from apps.stock_beta import Main as BETA
from apps.stock_members import Main as MBRS
from apps.stock_ta import Main as GP
from apps.stock_ATR import Main as ATR
from apps.stock_DVD_HK import Main as HK_DVD

# TODO:
# add user tracking https://github.com/jrieke/streamlit-analytics

def Main():
	st.set_page_config(
		layout = 'wide',
		page_title = 'Stocklit',
		page_icon = 'asset/app_logo_gold.png',
		initial_sidebar_state = 'expanded'
		)
	show_logo()
	with st.sidebar.expander("stocklit"):
		st.info(f'''
		[information symmetry](https://en.wikipedia.org/wiki/Information_asymmetry) for all

		*	[project page](https://github.com/ohjho/stocklit)
		*	[issues tracking](https://github.com/ohjho/stocklit/issues)
		''')

	app_dict = {
		"DESC": Desc,
		"GP": GP,
		"RT": RT,
		"ATR": ATR,
		"BETA": BETA,
		"MBRS": MBRS,
		"HK-DVD": HK_DVD,
		"login": login,
	}

	app_sw = st.sidebar.selectbox('select app', options = [''] + list(app_dict.keys()))
	if app_sw:
		app_func = app_dict[app_sw]
		app_func()

if __name__ == '__main__':
	Main()
