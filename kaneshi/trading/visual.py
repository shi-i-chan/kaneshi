import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from kaneshi.trading.database import DataBase
from kaneshi.config import DEF_PLOTLY_HEIGHT, DEF_PLOTLY_WIDTH

from typing import NoReturn


class RSIPlot:
    def __init__(self,
                 database_name: str,
                 bottom_edge: int = None,
                 upper_edge: int = None,
                 table_name: str = 'data',
                 show_last_n: int = 500,
                 time_step: str = 's',  # 's' or 'min'
                 chart_shift: int = 50):
        self.database = DataBase(database_name)
        self.BOTTOM_EDGE = bottom_edge
        self.UPPER_EDGE = upper_edge
        self.table_name = table_name
        self.show_last_n = show_last_n
        self.time_step = time_step
        self.chart_shift = chart_shift
        self.fig = None
        self.plot_data = None

    def init_plotting(self) -> NoReturn:
        """ Init data fot plotting """
        self.plot_data = self.database.read_table(self.table_name).tail(self.show_last_n)

        mirror = {'min': 'm', 's': 's'}
        last = self.plot_data['index'].values[-1]
        new_last = last + np.timedelta64(self.chart_shift, mirror[self.time_step])
        new_dates = pd.date_range(last, new_last, freq=f'1{self.time_step}')
        new_df = pd.DataFrame(new_dates, columns=['index'])
        self.plot_data = pd.concat([self.plot_data, new_df]).reset_index(drop=True)

    def plot_price(self) -> NoReturn:
        """ Plot price values to subplot """
        self.fig.add_trace(go.Scatter(x=self.plot_data['index'],
                                      y=self.plot_data['price'],
                                      line=dict(color='blue'),
                                      mode='lines',
                                      name='Price',
                                      ),
                           row=1,
                           col=1
                           )

    def plot_rsi(self) -> NoReturn:
        """ Plot RSI values to subplot """
        self.fig.add_trace(go.Scatter(x=self.plot_data['index'],
                                      y=self.plot_data['rsi'],
                                      line=dict(color='blue'),
                                      mode='lines',
                                      name='RSI',
                                      ),
                           row=2,
                           col=1
                           )

    def plot_edges(self) -> NoReturn:
        """ Plot RSI strategy edges to subplot """
        if self.BOTTOM_EDGE is not None:
            self.fig.add_hline(y=self.BOTTOM_EDGE, line_width=3, line_color='green', row=2)
        if self.UPPER_EDGE is not None:
            self.fig.add_hline(y=self.UPPER_EDGE, line_width=3, line_color='red', row=2)

    def plot_signals(self, kind: str, color: str, symbol: str) -> NoReturn:
        """ Plot specific signals to subplot """
        df = getattr(self, f'{kind}_orders_df')
        if df is not None:
            self.fig.add_trace(go.Scatter(x=df['index'],
                                          y=df['price'],
                                          mode="markers",
                                          legendgroup=f'{kind.capitalize()}',
                                          showlegend=True,
                                          marker=dict(
                                              symbol=symbol,
                                              color=color,
                                              size=12,
                                          ),
                                          name=f'{kind.capitalize()}',
                                          ),
                               row=1,
                               col=1
                               )

    def plot(self) -> NoReturn:
        """ Init figure and plot all charts """
        self.init_plotting()
        self.fig = make_subplots(rows=2, cols=1,
                                 shared_xaxes=True)
        self.plot_price()
        self.plot_rsi()
        self.plot_edges()

    def show(self) -> NoReturn:
        """ Plot and show figure """
        self.plot()
        self.fig.update_layout(
            height=DEF_PLOTLY_HEIGHT,
            width=DEF_PLOTLY_WIDTH,
            dragmode='pan',
            newshape_line_color='cyan',
            modebar_add=['drawline',
                         'eraseshape']
        )
        config = dict({'scrollZoom': True})
        self.fig.show(config=config)


class ClearRSIPlot(RSIPlot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_orders_data(self) -> NoReturn:
        """ Process orders data """
        for order_type in ['buy', 'sell']:
            setattr(self, f'{order_type}_orders_df',
                    self.plot_data[self.plot_data[f'{order_type}_order'] != 'None'][['index', 'price']].dropna())

    def init_plotting(self) -> NoReturn:
        """ Init data fot plotting """
        super().init_plotting()
        self.get_orders_data()

    def plot(self) -> NoReturn:
        """ Init figure and plot all charts """
        super().plot()
        self.plot_signals('buy', 'green', 'triangle-up')
        self.plot_signals('sell', 'red', 'triangle-down')
        return self.fig
