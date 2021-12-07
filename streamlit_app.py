import streamlit as st
import os, sys

from toolbox.st_utils import show_logo
from apps.stock_desc import Main as Desc
from apps.stock_returns import Main as RT
from apps.stock_beta import Main as BETA
from apps.stock_members import Main as MBRS
from apps.stock_ta import Main as TA
from apps.stock_ATR import Main as ATR
from apps.stock_DVD_HK import Main as HK_DVD

# TODO:
# add user tracking https://github.com/jrieke/streamlit-analytics

def Main():
	st.set_option('deprecation.showPyplotGlobalUse', False)
	st.set_page_config(layout = 'wide')
	show_logo()
	# st.sidebar.header('OpenTerminal')
	# st.sidebar.subheader('information symmetry for all')
	with st.sidebar.expander("OpenTerminal"):
		st.info(f'''
		[information symmetry](https://en.wikipedia.org/wiki/Information_asymmetry) for all

		*	[project page](https://github.com/ohjho/open_terminal)
		*	[issues tracking](https://github.com/ohjho/open_terminal/issues)
		''')

	app_dict = {
		"stock DESC": Desc,
		"stock RT": RT,
		"stock BETA": BETA,
		# "stock MBRS": MBRS,
		"stock TA": TA,
		"stock ATR": ATR,
		# "stock HK-DVD": HK_DVD
	}

	app_sw = st.sidebar.selectbox('select app', options = [''] + list(app_dict.keys()))
	if app_sw:
		app_func = app_dict[app_sw]
		app_func()

if __name__ == '__main__':
	Main()
