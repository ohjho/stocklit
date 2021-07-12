import os, sys, json
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def st_write_dict(data):
    for k, v in data.items():
        st.subheader(k)
        st.write(v)

def show_logo(st_asset = st.sidebar, use_column_width = False, width = 100):
    logo_url = "https://raw.githubusercontent.com/ohjho/open_terminal/master/asset/ot_logo.png"
    st_asset.image(logo_url, width = width, use_column_width = use_column_width)

def show_plotly(fig, height = None, title = None, template = 'plotly_dark', st_asset = st):
    params = {'height': height, 'title': title, 'template': template}
    params = {k:v for k,v in params.items() if v}

    fig.update_layout(params)
    st_asset.plotly_chart(fig, use_container_width = True, height = height)

def plotly_hist_draw_hline(fig, l_value_format):
    '''
    added a horizontial line in place
    Args:
        l_value_format: list of dictionary of the shape
            {value: 123, line_format: Optional[{'color': '#b58900', 'dash': 'dot', 'width': 1}]}

    ref: https://github.com/plotly/plotly_express/issues/143
    '''
    default_line_format = {'color': 'light grey', 'dash': 'dot', 'width': 1}
    l_shapes = []
    for shape in l_value_format:
        shape_dict = {'type': 'line',
                'yref': 'y', 'y0': shape['value'], 'y1': shape['value'],
                'xref': 'paper', 'x0': 0, 'x1': 1,
                'line': shape['line_format'] if 'line_format' in shape.keys() else default_line_format
            }
        # if line_format_params:
        #     shape_dict['line'] = line_format_params
        l_shapes.append(shape_dict)

    fig.update_layout(
        shapes = l_shapes
        )
