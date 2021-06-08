import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plotly_ohlc_chart(df, vol_col = None, date_col = None,
        ohlc_col_map = {'o':'Open', 'h':'High', 'l': 'Low', 'c':'Close'},
        show_legend = True
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

        # include a go.Bar trace for volumes
        fig.add_trace(go.Bar(x= date_serie, y= df[vol_col],
                        name = 'volume', showlegend = show_legend),
                       row = 2, col = 1)
        fig.update(layout_xaxis_rangeslider_visible=False)
    else:
        fig = go.Figure(data= go.Ohlc(x = date_serie,
                            open= df[ohlc_col_map['o']],
                            high= df[ohlc_col_map['h']],
                            low= df[ohlc_col_map['l']],
                            close= df[ohlc_col_map['c']])
                        )
    return fig
