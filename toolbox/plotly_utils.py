import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def get_moving_average_col(df_columns):
    return [c for c in df_columns if '_' in c and c.split('_')[0] in ['ema', 'sma', 'vwap']]

def get_channels_col(df_columns):
    return [c for c in df_columns if 'ch:' in c]

def add_channel_trace(fig, df, ch_cols, date_col = None, rgb_tup = (255,222,0)):
    # https://plotly.com/python/line-charts/
    # assume channels are always in pair and uses atr only (i.e. created by add_ATR())
    l_ch = [float(col.split('+')[-1].replace('atr',''))
            for col in ch_cols if '+' in col]
    l_ch = sorted(l_ch)[::-1] # descending
    date_serie = df[date_col] if date_col else df.index

    # to do color it the same color as the selected ema
    # ref: https://stackoverflow.com/questions/61353532/plotly-how-to-get-the-trace-color-attribute-in-order-to-plot-selected-marker-wi
    for i, ch in enumerate(l_ch):
        ch_hi, ch_lo = [c for c in ch_cols if str(ch) in c]
        fig.add_trace(go.Scatter(
                x = date_serie.tolist() + date_serie.tolist()[::-1],
                y = df[ch_lo].tolist() + df[ch_hi].tolist()[::-1],
                fill = "toself",
                fillcolor = f'rgba({rgb_tup[0]},{rgb_tup[1]},{rgb_tup[2]},{(i+1)/10.0})',
                line_color = 'rgba(255,255,255,0)',
                name = ch_hi
            ),
            row = 1,
            col = 1
        )

    return fig

def add_impulse_trace(fig, df, date_col = None,
        ohlc_col_map = {'o':'Open', 'h':'High', 'l': 'Low', 'c':'Close'},
        l_rgb_css_color_name = ['red', 'green', 'DodgerBlue']
        ):
    '''add candlestick trace on top to show impulse system
    ref: https://stackoverflow.com/a/66998861/14285096
    '''
    df_green = df[df['impulse'] == 1]
    df_red = df[df['impulse'] == -1]
    df_blue = df[df['impulse'] == 0]

    l_trace_def = [
        {'name': 'long or short', 'df': df_blue, 'color': l_rgb_css_color_name[2]},
        {'name': 'long only', 'df': df_green, 'color': l_rgb_css_color_name[1]},
        {'name': 'short only', 'df': df_red, 'color': l_rgb_css_color_name[0]},
    ]
    for trace in l_trace_def:
        df = trace['df']
        fig.add_trace(
            go.Candlestick( x = df[date_col] if date_col else df.index,
                open = df[ohlc_col_map['o']],
                high = df[ohlc_col_map['h']],
                low = df[ohlc_col_map['l']],
                close = df[ohlc_col_map['c']],
                name = trace['name'],
                increasing_line_color = trace['color'],
                decreasing_line_color = trace['color']
                ),
            row = 1, col = 1
        )
    return fig

def add_volume_profile(fig, df, vol_col, price_col, show_legend = False,
        color = 'Ivory', opacity = 0.6):
    '''add volume profile to candlestick chart
    add a go.Bar trace for volume-at-price
    ref: https://stackoverflow.com/questions/58517234/using-a-charting-library-to-overlay-volume-profile-on-a-candlestick-chart-in-pyt
    '''
    # TODO: bug fix, volume profile will disable volume on row 2
    df_vp = df[[price_col, vol_col]].groupby(price_col).sum()
    fig.update_layout(
        xaxis2 = go.layout.XAxis(side = 'top',
                    range = [0,max(df_vp[vol_col])], overlaying = 'x', anchor = 'y',
                    rangeslider = go.layout.xaxis.Rangeslider(visible = False),
                    showticklabels = False),
        # yaxis2 = go.layout.YAxis(side = 'left',
        #             range = [df[price_col].min(), df[price_col].max()],
        #             overlaying = 'y')
        )
    fig.add_trace(
        go.Bar(y = df_vp.index, x = df_vp[vol_col], orientation = 'h',
            name = 'volume profile', showlegend = show_legend,
            xaxis = 'x2', yaxis = 'y',
            marker_color = color, opacity = opacity),
        #secondary_x = True,
        row = 1, col = 1)
    # fig.data[1].update(xaxis = 'x2')
    return fig

def plotly_ohlc_chart(df, vol_col = None, date_col = None, show_volume_profile = False,
        ohlc_col_map = {'o':'Open', 'h':'High', 'l': 'Low', 'c':'Close'},
        show_legend = True, show_range_slider = True
    ):
    '''
    return a fig object with a ohlc plot
    reference this stackoverflow solution: https://stackoverflow.com/a/65997291/14285096
    Args:
        show_range_slider: only applies if vol_col is None
    '''
    date_serie = df[date_col] if date_col else df.index
    if vol_col:
        row_count = 2
        row_count += 1 if "MACD_histogram" in df.columns else 0
        row_count += 1 if "A/D" in df.columns else 0
        row_count += 1 if "OBV" in df.columns else 0
        row_heights = [0.7] + [0.2 for i in range(row_count-1)]
        # Create figure with secondary y-axis
        fig = make_subplots(rows = row_count, cols = 1, shared_xaxes= True,
                vertical_spacing= 0.03,
                subplot_titles = ['OHLC','Volume'] if not show_legend else None,
                row_heights = row_heights)



        # include candlestick with rangeselector
        fig.add_trace(go.Candlestick(x= date_serie,
                        open= df[ohlc_col_map['o']],
                        high= df[ohlc_col_map['h']],
                        low= df[ohlc_col_map['l']],
                        close= df[ohlc_col_map['c']],
                        name = 'OHLC', showlegend = show_legend),
                        row = 1, col =1
                        )

        if show_volume_profile:
            fig = add_volume_profile(fig, df, vol_col = vol_col,
                    price_col = ohlc_col_map['c'], show_legend = show_legend)

        # include a go.Bar trace for volumes
        fig.add_trace(go.Bar(x= date_serie, y= df[vol_col],
                        name = 'volume', showlegend = show_legend),
                       row = 2, col = 1)
        fig.update(layout_xaxis_rangeslider_visible=False)
    else:
        fig = go.Figure(data= go.Candlestick(x = date_serie,
                            open= df[ohlc_col_map['o']],
                            high= df[ohlc_col_map['h']],
                            low= df[ohlc_col_map['l']],
                            close= df[ohlc_col_map['c']])
                        )
        if not show_range_slider:
            fig.update_layout(xaxis_rangeslider_visible=False)

    # check for TA to add
    if 'impulse' in df.columns:
        fig = add_impulse_trace(fig, df, ohlc_col_map = ohlc_col_map, date_col = date_col)

    ma_cols = get_moving_average_col(df.columns)
    if len(ma_cols)> 0:
        for ma in ma_cols:
            fig.add_trace(go.Scatter(x = date_serie, y = df[ma], name = ma),
                    row = 1, col = 1 )
    ch_cols = get_channels_col(df.columns)
    if len(ch_cols)> 0:
        fig = add_channel_trace(fig, df, ch_cols = ch_cols, date_col = date_col,
                rgb_tup = (231,107,243))

    # Check for additional TA subplots
    ref_row = 2 if vol_col else 1
    if 'MACD_histogram' in df.columns:
        ref_row += 1
        fig.append_trace(go.Bar(x = date_serie, y = df['MACD_histogram'], name = 'MACD_histogram'),
            row = ref_row, col = 1)
        # fig.add_trace(go.Scatter(x = df.index, y = df['MACD_signal'], name = 'MACD_signal'),
        #     row = 3, col = 1)

    if 'A/D' in df.columns:
        ref_row += 1
        fig.append_trace(go.Scatter(x = date_serie, y = df['A/D'], name = 'A/D'),
            row = ref_row, col = 1)

    if 'OBV' in df.columns:
        ref_row += 1
        fig.append_trace(go.Scatter(x = date_serie, y = df['OBV'], name = 'OBV'),
            row = ref_row, col = 1)

    #TODO: hide range outside trading hours: https://stackoverflow.com/a/65632833/14285096
    return fig
