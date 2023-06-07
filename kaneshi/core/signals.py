import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objects import Figure

from kaneshi.core.indicators import DefaultIndicator

from numpy.typing import NDArray
from typing import TypeVar, List, NoReturn, Tuple
Indicator = TypeVar('Indicator', bound=DefaultIndicator)


class DefaultSignal:
    def __init__(self,
                 kind: str = None,
                 minor_indicator: Indicator = None,
                 major_indicator: Indicator = None,
                 ):
        self.kind = kind
        self.minor_indicator = minor_indicator
        self.major_indicator = major_indicator
        self.plus_one = ['up', 'more']
        self.minus_one = ['down', 'less']

    def calc_positions(self, *args, **kwargs):
        """ Calculate positions for specific signal generator """
        raise NotImplementedError

    def calc_indicators(self, price_array: NDArray) -> NoReturn:
        """ Apply indicators to price """
        self.minor_indicator.apply_to(price_array=price_array)
        self.major_indicator.apply_to(price_array=price_array)

    def get_signals(self, price: NDArray, *args, **kwargs) -> NDArray:  # NOQA
        """ Get signals from positions """
        positions = self.calc_positions(price)
        mask = []
        if self.kind in self.plus_one:
            mask = positions == 1
        elif self.kind in self.minus_one:
            mask = positions == -1
        elif self.kind == 'any':
            mask = np.any(np.stack([positions == 1, positions == -1], axis=0), axis=0)  # TODO test
        return mask


class Crossover(DefaultSignal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calc_positions(self, price_array: NDArray) -> NDArray:
        """ Calc positions where minor indicator cross major indicator up or down """
        super().calc_indicators(price_array)
        signals = np.where(self.minor_indicator.values > self.major_indicator.values, 1, 0)
        signals = np.insert(np.diff(signals), 0, 0)  # TODO why shifted?
        return signals


class StopsMixin:
    def get_sell_index(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def resampler(values: List[float], indices: List[np.datetime64]) -> Tuple[NDArray, NDArray]:
        """ Resample collected stop values """
        s = pd.Series(values, name='values', index=indices)
        s = s.resample('1min').asfreq()
        return s.values, s.index.values

    @staticmethod
    def plotter(fig: Figure, row_number: int, values: NDArray, indices: NDArray, color: str, name: str) -> NoReturn:
        """ Plot values array to subplot """
        fig.add_trace(go.Scatter(x=np.array(indices),
                                 y=np.array(values),
                                 line=dict(color=color),
                                 mode='lines',
                                 name=name,
                                 ),
                      row=row_number,
                      col=1
                      )

    def get_signals(self, price_array: NDArray, index_array: NDArray, buy_indices: NDArray) -> Tuple[NDArray, NDArray]:
        """ Calculate stop/take signals for all raw buy and sell indices """
        cleaned_buy_indices, sell_indices = [], []
        last_sell_index = 0
        for buy_index in buy_indices:
            if buy_index > last_sell_index:
                sell_index = self.get_sell_index(price_array[buy_index:], index_array[buy_index:]) + buy_index
                cleaned_buy_indices.append(buy_index)
                sell_indices.append(sell_index)
                last_sell_index = sell_index
        return np.array(cleaned_buy_indices), np.array(sell_indices)


class DetStopTake(StopsMixin):
    def __init__(self, stop_loss_percent: float = 0, take_profit_percent: float = 0):
        self.stop_loss_percent = stop_loss_percent
        self.take_profit_percent = take_profit_percent

        self.indices = []
        self.take_values = []
        self.stop_values = []

    def init_plotting(self, *args, **kwargs):  # NOQA
        """ Initialize indices and values for plotting """
        self.take_values, _ = self.resampler(self.take_values, self.indices)
        self.stop_values, self.indices = self.resampler(self.stop_values, self.indices)
        return self

    def plot_to_plotly(self, fig: Figure, row_number: int) -> NoReturn:
        """ Plot indicator values to plotly subplot """
        self.plotter(fig, row_number, self.take_values, self.indices, 'green', 'Take profit values')  # NOQA
        self.plotter(fig, row_number, self.stop_values, self.indices, 'red', 'Stop loss values')  # NOQA
        fig.update_traces(connectgaps=False)

    def save_for_plot(self, stop_value: float, take_value: float, index: np.datetime64) -> NoReturn:
        """ Save stop_values and indices to plot """
        self.stop_values.append(stop_value)
        self.take_values.append(take_value)
        self.indices.append(index)

    def get_sell_index(self, price_array: NDArray, indices_array: NDArray) -> int:
        """ Calculate adaptive stop sell index for current buy_index and buy_price """
        stop_loss_price = price_array[0] * (1 + self.stop_loss_percent) if self.stop_loss_percent != 0 else np.nan
        take_profit_price = price_array[0] * (1 + self.take_profit_percent) if self.take_profit_percent != 0 else np.nan

        for price_index, price in enumerate(price_array):
            if price < stop_loss_price or price > take_profit_price:
                return price_index

            if price_index < len(indices_array) - 1:
                self.save_for_plot(stop_loss_price, take_profit_price, indices_array[price_index + 1])
        return len(price_array) - 1
