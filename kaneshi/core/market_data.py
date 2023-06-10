import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from kaneshi.utils import read_df, ds_to_h5
from kaneshi.config import DEF_MARKET_DATA_PATH, DEF_PLT_FIGSIZE, DEF_PLOTLY_HEIGHT, DEF_PLOTLY_WIDTH

from numpy.typing import NDArray
from typing import List, Union, NoReturn, Tuple


class MarketData:
    def __init__(self,
                 raw_df: pd.DataFrame = None,
                 symbol: str = 'symbol',
                 interval: int = 1,
                 s_date: Tuple = None,
                 e_date: Tuple = None,
                 columns: Union[List[str], str] = None,
                 price_type: str = 'Close',
                 ):
        self.raw_df = raw_df
        self.symbol = symbol
        self.interval = interval
        self.s_date = s_date
        self.e_date = e_date
        self.columns = None if columns is None else columns if isinstance(columns, list) else [columns]
        self.price_type = price_type

        self.dataframes = None
        self.current_df = None
        self.current_price = None
        self.current_index = None

        self.__prepare_dataframes()

    @classmethod
    def from_config(cls,
                    symbol: str,
                    interval: int = 1,
                    s_date: Tuple = None,
                    e_date: Tuple = None,
                    columns: Union[List[str], str] = None,
                    price_type: str = 'Close') -> 'MarketData':
        """ Initialize market data from file """
        raw_df = read_df(rf'{DEF_MARKET_DATA_PATH}\raw_{symbol}_1m')
        return cls(raw_df=raw_df, symbol=symbol, interval=interval,
                   s_date=s_date, e_date=e_date, columns=columns, price_type=price_type)

    # ONLY FOR MINUTES
    def dataset_from_indices(self, lookback: int, good_indices: NDArray,
                             bad_indices: NDArray, dataset_fn: str = None) -> NoReturn:
        """ Create dataset from buy indices """
        examples, labels, indices = [[] for _ in range(3)]
        buy_indices = np.concatenate([good_indices, bad_indices])
        for end_index in buy_indices:
            start_index = end_index - np.timedelta64(lookback - 1, 'm')
            if start_index in self.raw_df.index and end_index in self.raw_df.index:
                label = 1 if end_index in good_indices else 0 if end_index in bad_indices else None
                example = self.raw_df[start_index: end_index].values
                if example.shape[0] == lookback:
                    labels.append(label)
                    examples.append(example)
                    indices.append(end_index)
        data = {'examples': np.array(examples),
                'labels': np.array(labels),
                'indices': np.array(indices).astype(np.int64)}
        ds_to_h5(dataset_fn, data)

    def get_price(self) -> NDArray:
        """ Get price from full dataframe """
        return self.raw_df[self.price_type].values

    def get_price_by_indices(self, indices: NDArray[np.datetime64]) -> NDArray[np.float32]:
        """ Return price by datetime indices """
        return self.raw_df.loc[indices][self.price_type].values

    def resample_df(self, interval: int) -> 'MarketData':
        """ Resample price dataframe with minutes interval """
        ohlc = {'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum',
                'Quote asset volume': 'sum',
                'Number of trades': 'sum',
                'Taker buy base asset volume': 'sum',
                'Taker buy quote asset volume': 'sum',
                }
        columns = self.raw_df.columns.tolist()
        ohlc = dict((key, ohlc[key]) for key in columns if key in ohlc)
        self.raw_df = self.raw_df.resample(f'{interval}min').apply(ohlc)
        self.raw_df = self.raw_df.dropna()
        return self

    def plot_lines(self, columns: List[str] = None, type_: str = 'plotly') -> NoReturn:
        """ Plot selected columns as lines """
        columns = columns if columns is not None else ['Close']
        getattr(self, f'plot_lines_{type_}')(columns)

    def plot_lines_plotly(self, columns: List[str]) -> NoReturn:
        """ Plot lines with plotly """
        fig = go.Figure()
        for col in columns:
            fig.add_trace(go.Scatter(x=self.raw_df.index, y=self.raw_df[col], name=col))
        fig.update_layout(
            height=DEF_PLOTLY_HEIGHT,
            width=DEF_PLOTLY_WIDTH,
            dragmode='pan',
            newshape_line_color='cyan',
            modebar_add=[
                'drawline',
                'eraseshape'
            ],
            title="Price chart",
            xaxis_title="Date",
            yaxis_title="Price",
            showlegend=True,
        )
        fig_config = dict({'scrollZoom': True})
        fig.show(config=fig_config)

    def plot_lines_plt(self, columns: List[str]) -> NoReturn:
        """ Plot lines with plt """
        plt.figure(figsize=DEF_PLT_FIGSIZE)
        for col in columns:
            plt.plot(self.raw_df.index, self.raw_df[col], label=col)
        plt.legend(loc='upper left')
        plt.title('Price chart')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.tight_layout()
        plt.show()

    def __iter__(self) -> 'MarketData':
        """ Init market data iterator """
        self.df_iterator = iter(self.dataframes)
        return self

    def __next__(self) -> NoReturn:
        """ Iterate market data. Iter to next dataframe and reset some variables """
        self.current_df = next(self.df_iterator)
        self.current_price = self.current_df[self.price_type].values
        self.current_index = self.current_df.index.values

    @staticmethod
    def __select_date_range(df: pd.DataFrame, start_date: Tuple[int], end_date: Tuple[int]) -> pd.DataFrame:
        """ Crop dates only between two years """
        if start_date is not None and end_date is not None:
            start_date, end_date = datetime(*start_date), datetime(*end_date)
            mask = (df.index >= start_date) & (df.index < end_date)
            return df.loc[mask]
        return df

    def __split_df(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """ Split raw dataframe to chunks by gaps """
        df['identifier'] = (~df.index.to_series().diff().dt.seconds.div(self.interval * 60,
                                                                        fill_value=0).lt(2)).cumsum()
        return [df.drop(columns=['identifier']) for _, df in df.groupby('identifier')]

    def __prepare_dataframes(self) -> NoReturn:
        """ Prepare raw price dataframe to process. Cut, split, etc """
        self.raw_df.index = pd.to_datetime(self.raw_df.index.values)
        if self.interval != 1:
            self.resample_df(interval=self.interval)
        self.raw_df = self.raw_df[self.columns] if self.columns is not None else self.raw_df
        self.raw_df = self.__select_date_range(self.raw_df, self.s_date, self.e_date)
        self.dataframes = self.__split_df(self.raw_df.copy())
