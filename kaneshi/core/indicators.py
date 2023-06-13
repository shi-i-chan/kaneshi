import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objects import Figure

from typing import NoReturn
from numpy.typing import NDArray


class DefaultIndicator:
    def __init__(self, color: str = 'blue', label: str = None):
        self.df = None
        self.values = None
        self.color = color
        self.label = label

    def apply_to(self, *args, **kwargs):
        raise NotImplementedError

    def init_plotting(self, df: pd.DataFrame):  # TODO add return
        """ Initialize data for plotting """
        if self.df is None:
            df['values'] = self.apply_to(df['Close'])
            self.df = df
        return self

    def get_values_by_indices(self, indices: NDArray[np.datetime64]) -> NDArray[np.float32]:
        """ Get values from initialized dataframe by datetime indices """
        return self.df.loc[indices]['values']

    def plot_to_plotly(self, fig: Figure, row_number: int) -> NoReturn:
        """ Plot indicator values to plotly subplot """
        fig.add_trace(go.Scatter(x=self.df.index,
                                 y=self.df['values'],
                                 line=dict(color=self.color),
                                 mode='lines',
                                 name=self.label,
                                 ),
                      row=row_number,
                      col=1
                      )


class Price(DefaultIndicator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = 'Price' if self.label is None else self.label

    def apply_to(self, price_array: NDArray) -> NDArray:
        """ Apply indicator to price. In this case just return price """
        self.values = price_array
        return self.values


class Constant(DefaultIndicator):
    def __init__(self, constant: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constant = constant
        self.label = f'Constant {constant}' if self.label is None else self.label

    def plot_to_plotly(self, fig: Figure, row_number: int) -> NoReturn:
        """ Plot indicator to plotly. In this case just horizontal line """
        if self.constant is not None:
            fig.add_hline(y=self.constant, line_width=3, line_color=self.color, row=row_number)

    def apply_to(self, price_array: NDArray) -> NDArray:
        """ Apply indicator to price array. In this case just make constant array with same shape """
        self.values = np.full_like(price_array, self.constant)
        return self.values


class MA(DefaultIndicator):
    def __init__(self, ma_window: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ma_window = ma_window


class Simple_MA(MA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = f'SMA_{self.ma_window}' if self.label is None else self.label

    def apply_to(self, price_array: NDArray) -> NDArray:
        """ Apply indicator to price array. In this case calc moving average """
        price_series = pd.Series(price_array)
        self.values = price_series.rolling(window=self.ma_window).mean().values
        return self.values


class RSI(DefaultIndicator):  # based on EMA and SMA
    def __init__(self, period: int, ema: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.period = period
        self.ema = ema
        self.label = f'RSI_{period}' if self.label is None else self.label

    def apply_to(self, price_array: NDArray) -> NDArray:
        """ Apply indicator to price array. In this case calc RSI """
        price_series = pd.Series(price_array)
        close_delta = price_series.diff()
        up = close_delta.clip(lower=0)
        down = -1 * close_delta.clip(upper=0)

        if self.ema:  # EMA
            ma_up = up.ewm(com=self.period - 1, adjust=True, min_periods=self.period).mean()
            ma_down = down.ewm(com=self.period - 1, adjust=True, min_periods=self.period).mean()
        else:  # SMA
            ma_up = up.rolling(window=self.period).mean()
            ma_down = down.rolling(window=self.period).mean()

        rsi = ma_up / ma_down
        rsi = 100 - (100 / (1 + rsi))
        self.values = rsi.values
        return self.values


class PercentChange(DefaultIndicator):
    def __init__(self, period: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.periods = period
        self.label = f'Percent change {period}' if self.label is None else self.label

    def apply_to(self, price_array: NDArray) -> NDArray:
        """ Apply indicator to price array. In this case calc percent change by period """
        price_series = pd.Series(price_array)
        pct_change = price_series.pct_change(periods=self.periods).values
        self.values = pct_change * 100
        return self.values


