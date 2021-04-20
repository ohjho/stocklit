import os, sys, json
import numpy as np
import streamlit as st

def st_write_dict(data):
    for k, v in data.items():
        st.subheader(k)
        st.write(v)

def show_logo(st_asset = st.sidebar, use_column_width = False, width = 100):
    logo_url = 'ot_logo.png'
    st_asset.image(logo_url, width = width, use_column_width = use_column_width)
