import streamlit as st
import os, sys
import streamlit_authenticator as stauth

#Paths
cwdir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, os.path.join(cwdir, "../"))
from toolbox.st_auth import interactive_login

def Main():
    interactive_login(st_asset = st.container(), b_show_logo = True)

if __name__ == '__main__':
    Main()
