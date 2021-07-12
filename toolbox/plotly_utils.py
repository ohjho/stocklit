import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plotly_ohlc_chart(df, vol_col = None, date_col = None, show_volume_profile = False,
        ohlc_col_map = {'o':'Open', 'h':'High', 'l': 'Low', 'c':'Close'},
        show_legend = True, show_range_slider = True
    ):
    '''
    return a fig object with a ohlc plot
    reference this stackoverflow solution: https://stackoverflow.com/a/65997291/14285096
    '''
    date_serie = df[date_col] if date_col else df.index
    if vol_col:
        # Create figure with secondary y-axis
        fig = make_subplots(rows = 2, cols = 1, shared_xaxes= True,
                vertical_spacing= 0.03,
                subplot_titles = ['OHLC','Volume'] if not show_legend else None,
                row_width = [0.2,0.7])



        # include candlestick with rangeselector
        fig.add_trace(go.Candlestick(x= date_serie,
                        open= df[ohlc_col_map['o']],
                        high= df[ohlc_col_map['h']],
                        low= df[ohlc_col_map['l']],
                        close= df[ohlc_col_map['c']],
                        name = 'OHLC', showlegend = show_legend),
                        row = 1, col =1
                        )

        # include a go.Bar trace for volume-at-price
        # TODO: bug fix, volume profile will disable volume on row 2
        # ref: https://stackoverflow.com/questions/58517234/using-a-charting-library-to-overlay-volume-profile-on-a-candlestick-chart-in-pyt
        if show_volume_profile and vol_col:
            price_col = ohlc_col_map['c']
            df_vp = df[[price_col, vol_col]].groupby(price_col).sum()
            fig.update_layout(xaxis2 = go.layout.XAxis(side = 'top', range = [0,max(df_vp[vol_col])], overlaying = 'x', anchor = 'y'))
            fig.add_trace(go.Bar(y = df_vp.index, x = df_vp[vol_col], orientation = 'h', name = 'volume profile', showlegend = show_legend),
                #secondary_x = True,
                row = 1, col = 1)
            fig.data[1].update(xaxis = 'x2')

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

    #TODO: hide range outside trading hours: https://stackoverflow.com/a/65632833/14285096
    return fig
