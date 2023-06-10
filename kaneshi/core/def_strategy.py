import os
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

from kaneshi import utils
from kaneshi import config
from kaneshi.core import metrics, signals
from kaneshi.core.indicators import Price
from kaneshi.core.signals import DefaultSignal
from kaneshi.core.market_data import MarketData

from numpy.typing import NDArray
from typing import Union, NoReturn, Tuple, Iterable


class StrategyVariables:
    def __init__(self):
        self.market_data = None
        self.buy_conditions = None
        self.sell_conditions = None
        self.buy_indices = None
        self.sell_indices = None
        self.stop = None
        self.trades = None
        self.plot_config = None
        self.report_config = None
        self.commission = None
        self.real_sell_prices = None

    def __init_bs_indices(self, *args, **kwargs):
        raise NotImplementedError

    def _get_price_by_indices(self, *args, **kwargs):
        raise NotImplementedError

    def clear_indices(self, *args, **kwargs):
        raise NotImplementedError

    def _get_trades(self, *args, **kwargs):
        raise NotImplementedError

    def _bs_indices_to_dt(self, *args, **kwargs):
        raise NotImplementedError

    def _apply_to_sub_df(self, *args, **kwargs):
        raise NotImplementedError


class StrategyDatasetsMixin(StrategyVariables):
    def create_dataset(self, lookback: int, dataset_fn: str) -> NoReturn:
        """ Create dataset for current strategy and market data """
        dataset_fn = f'{config.DEF_DATASETS_PATH}/{dataset_fn}.h5'
        good_buy_idx, bad_buy_idx = self.__get_good_bad_buy_indices()
        self.market_data.dataset_from_indices(lookback, good_buy_idx, bad_buy_idx, dataset_fn=dataset_fn)

    def __get_good_bad_buy_indices(self) -> Tuple[NDArray[np.datetime64], ...]:
        """ Get good and bad buy indices """
        good_buy_indices = self.buy_indices[self.trades > 0]
        bad_buy_indices = self.buy_indices[self.trades < 0]
        return good_buy_indices, bad_buy_indices


class StrategyPlotting(StrategyVariables):
    def __init__(self, ):
        super().__init__()
        index_types = ['buy', 'sell']
        index_colors = ['green', 'red']
        index_symbols = ['triangle-up', 'triangle-down']
        self.indices_data = {index_type: {'n_plots': 0, 'color': index_color, 'symbol': index_symbol}
                             for index_type, index_color, index_symbol in zip(index_types, index_colors, index_symbols)}

    def plot(self, plot_type: str = 'plotly') -> Union[NoReturn, go.Figure]:
        """ Plot and show or return plotly figure according to plot_config """
        nrows = len(self.plot_config)
        fig = make_subplots(rows=nrows, cols=1,
                            shared_xaxes=True)
        for row_idx in range(1, nrows + 1):
            self.__plot_plotly_subplot(fig, row_idx, self.plot_config[row_idx - 1])

        fig.update_layout(
            height=config.DEF_PLOTLY_HEIGHT,
            width=config.DEF_PLOTLY_WIDTH,
            dragmode='pan',
            newshape_line_color='cyan',
            modebar_add=['drawline',
                         'eraseshape'
                         ]
        )
        fig_config = dict({'scrollZoom': True})
        return utils.process_fig(fig, fig_config, plot_type)

    def plot_return(self, plot_type: str = 'plotly', dep_type: str = 'cumulative') -> Union[NoReturn, NDArray[float]]:
        """ Plot return value and trades sign charts """
        fig = make_subplots(2, 1, shared_xaxes=True,
                            subplot_titles=('Return vs trades', 'Trades sign'))
        deposit = getattr(self, f'_{dep_type}_deposit')()
        fig.add_trace(go.Scatter(y=deposit, name='Return', mode='lines'), row=1, col=1)

        good_segments = self.__indices_to_segments(np.argwhere(self.trades > 0).flatten())
        bad_segments = self.__indices_to_segments(np.argwhere(self.trades < 0).flatten())
        self.__plot_segments(fig, good_segments, bad_segments, 2, 1)

        fig.update_yaxes(title_text="Return, percent", row=1, col=1)
        fig['layout']['yaxis2']['showticklabels'] = False
        fig['layout']['xaxis2']['title'] = 'Trade number'
        fig.update_layout(
            height=config.DEF_PLOTLY_HEIGHT,
            width=config.DEF_PLOTLY_WIDTH,
            dragmode='pan',
            newshape_line_color='cyan',
            modebar_add=[
                'drawline',
                'eraseshape'
            ]
        )
        fig_config = dict({'scrollZoom': True})
        return utils.process_fig(fig, fig_config, plot_type)

    def __plot_plotly_subplot(self, fig: Figure, row_number: int, indicators: dict) -> NoReturn:
        """ Fill specific subplot """
        major_indicator = indicators['major']
        major_indicator.init_plotting(self.market_data.raw_df).plot_to_plotly(fig, row_number)
        for index_type, index_data in self.indices_data.items():
            if self.real_sell_prices is not None and index_type == 'sell' and isinstance(major_indicator, Price):
                y_values = self.real_sell_prices
            else:
                y_values = None
            self.__plot_points(fig, row_number, major_indicator, index_data['color'],
                               index_type, symbol=index_data['symbol'], y_values=y_values)

        if 'minor' in indicators:
            for minor_indicator in indicators['minor']:
                minor_indicator.init_plotting(self.market_data.raw_df).plot_to_plotly(fig, row_number)

    def __plot_points(self, fig: Figure, row_number: int, major_indicator,
                      color: str, kind: str, symbol: str = 'triangle-up', y_values: NDArray = None) -> NoReturn:
        """ Plot buy sell signals to figure """
        if (indices := getattr(self, f'{kind}_indices')) is not None:
            self.indices_data[kind]['n_plots'] += 1
            fig.add_trace(go.Scatter(x=indices,
                                     y=major_indicator.get_values_by_indices(indices) if y_values is None else y_values,
                                     mode="markers",
                                     legendgroup=f'{kind.capitalize()}',
                                     showlegend=True if self.indices_data[kind]['n_plots'] < 2 else False,
                                     marker=dict(
                                         symbol=symbol,
                                         color=color,
                                         size=12,
                                     ),
                                     name=f'{kind.capitalize()}',
                                     ),
                          row=row_number,
                          col=1
                          )

    @staticmethod
    def __indices_to_segments(indices: NDArray[int]) -> NDArray[Tuple[int, ...]]:
        """ Convert indices array to array of tuples to fill areas """
        segments = []
        if len(indices) >= 1:
            if len(indices) == 1:
                return np.array([(indices[0], indices[0])])
            start_value = indices[0]
            for current_index, current_value in enumerate(indices[:-1]):
                next_value = indices[current_index + 1]
                if start_value is None:
                    start_value = current_value
                if next_value == current_value + 1:
                    pass  # ???
                else:
                    end_value = current_value
                    segments.append((start_value, end_value))
                    start_value = None

                if current_index == len(indices) - 2:
                    if start_value is not None:
                        segments.append((start_value, indices[-1]))
                    else:
                        segments.append((indices[-1], indices[-1]))
            return np.array(segments)
        else:
            return np.array(indices)

    def _cumulative_deposit(self) -> NDArray[int]:  # DRY
        """ Get deposit changing chart with cumulative entry into position """
        deps_chart = []
        dep = 100
        for trade in self.trades:
            dep = dep * (1 - config.DEF_COMMISSION)
            dep = dep + dep * float(trade)
            dep = dep * (1 - config.DEF_COMMISSION)
            deps_chart.append(dep)
        return np.array(deps_chart) - 100

    def _equal_deposit(self) -> NDArray[int]:  # DRY
        """ Get deposit changing chart with equal entry into position"""
        trades = self.trades
        start_dep = 100
        dep = np.full_like(trades, start_dep)
        dep = dep * (1 - config.DEF_COMMISSION)
        dep = dep + dep * trades
        dep = dep * (1 - config.DEF_COMMISSION)
        dep = dep - start_dep
        return np.cumsum(dep).tolist()

    @staticmethod
    def __plot_segments(fig: go.Figure, good_segments: NDArray[Tuple[int, ...]], bad_segments: NDArray[Tuple[int, ...]],
                        row: int, col: int) -> NoReturn:
        """ Plot segments to subplot """
        opacity = 0.5
        for coord in list(good_segments) + list(bad_segments):
            fillcolor = f'rgba(0, 255, 0, {opacity})' if coord in good_segments else f'rgba(255, 0, 0, {opacity})'
            x0, x1 = coord
            x0 -= 0.5
            x1 += 0.5
            seg_x = [x0, x1]
            seg_y = [1 for _ in range(len(seg_x))]
            fig.add_trace(go.Scatter(x=seg_x,
                                     y=seg_y,
                                     fill='tozeroy',
                                     fillcolor=fillcolor,
                                     showlegend=False,
                                     mode='none',
                                     ),
                          row=row,
                          col=col)


class Strategy(StrategyPlotting, StrategyDatasetsMixin):
    def __init__(self, market_data: MarketData = None):
        super().__init__()
        self.market_data = market_data
        self.commission = config.DEF_COMMISSION
        self.report_config = config.DEF_REPORT_CONFIG

    def apply(self) -> NoReturn:
        """ Apply strategy to all sub dfs and calc trades results """
        buy_indices, sell_indices = config.EMPTY_DT_ARRAY, config.EMPTY_DT_ARRAY
        for idx, _ in enumerate(iter(self.market_data)):
            self.__init_bs_indices()
            self._apply_to_sub_df()
            self._bs_indices_to_dt()
            buy_indices = np.concatenate([buy_indices, self.buy_indices])
            sell_indices = np.concatenate([sell_indices, self.sell_indices])

        self.buy_indices, self.sell_indices = buy_indices, sell_indices
        self._get_trades()
        return self

    def generate_report(self, type_: str = 'list') -> pd.DataFrame:
        """ Generate report from strategy results """
        if self.trades is not None:
            result = [round(getattr(metrics, metric_name)(self), config.DEF_ROUND)
                      for metric_name in self.report_config]
        else:
            result = [-1000 for _ in range(len(self.report_config))]
        return result if type_ == 'list' else pd.DataFrame([result], columns=self.report_config)

    def _get_trades(self) -> NoReturn:
        """ Calc array of trades results """
        if self.buy_indices.shape[0] != 0:
            buy_price = self._get_price_by_indices(self.buy_indices)
            sell_price = self._get_price_by_indices(self.sell_indices)
            self.trades = sell_price / buy_price - 1

    @staticmethod
    def _fix_bug_with_first_signals(raw_indices: NDArray[np.int32]) -> NDArray[np.int32]:
        return raw_indices

    def _get_price_by_indices(self, indices: NDArray[np.datetime64]) -> NDArray[np.float32]:
        """ Return price by datetime indices """
        return self.market_data.get_price_by_indices(indices)

    def _bs_indices_to_dt(self) -> NoReturn:
        """ Convert int signal indices to datetime """
        if self.buy_indices.shape[0] > 0:
            self.buy_indices = self.market_data.current_index[self.buy_indices.astype(int)]
            self.sell_indices = self.market_data.current_index[self.sell_indices.astype(int)]
        else:
            self.buy_indices = self.buy_indices.astype(dtype='datetime64')
            self.sell_indices = self.sell_indices.astype(dtype='datetime64')

    def _get_signal_indices(self, kind: str) -> Union[NDArray, bool]:
        """ Calculate and return signals indices or False """
        mask = self.__apply_conditions(self.market_data.current_price, **getattr(self, f'{kind}_conditions'))
        raw_indices = np.where(mask)[0]
        cleaned_indices = self._fix_bug_with_first_signals(raw_indices)
        if cleaned_indices.size != 0:
            return cleaned_indices
        return False

    def _add_stops_to_plot(self) -> NoReturn:
        """ Add stop signals generator to plot config """
        if 'minor' in self.plot_config[0]:
            self.plot_config[0]['minor'].append(self.stop)
        else:
            self.plot_config[0]['minor'] = [self.stop]

    def __init_bs_indices(self) -> NoReturn:
        """ Init buy and sell indices as empty arrays """
        self.buy_indices, self.sell_indices = config.EMPTY_DT_ARRAY, config.EMPTY_DT_ARRAY

    @staticmethod
    def __apply_conditions(price_array: NDArray,
                           signal_generators: Iterable[DefaultSignal],
                           merge_type: str = 'and',
                           ) -> NDArray[bool]:
        """ Apply buy and sell conditions """
        masks = list(map(lambda generator: generator.get_signals(price_array), signal_generators))
        masks = np.stack(masks, axis=0)
        mirror = {'and': 'all', 'or': 'any'}
        result_mask = getattr(np, mirror[merge_type])(masks, axis=0)
        return result_mask


class ClearStrategy(Strategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _apply_to_sub_df(self) -> NoReturn:
        """ Apply strategy to sub df and clear indices """
        if (buy_indices := self._get_signal_indices('buy')) is not False \
                and (sell_indices := self._get_signal_indices('sell')) is not False:
            cleaned_buy_indices, cleaned_sell_indices = [], []
            last_sell_index_index, last_sell_index_value = 0, 0
            for buy_index_value in buy_indices:
                if buy_index_value > last_sell_index_value:
                    for sell_index_index, sell_index_value in enumerate(sell_indices[last_sell_index_index:]):
                        if sell_index_value > buy_index_value:
                            cleaned_sell_indices.append(sell_index_value)
                            cleaned_buy_indices.append(buy_index_value)

                            last_sell_index_value = sell_index_value
                            last_sell_index_index += sell_index_index
                            break

            self.buy_indices = np.array(cleaned_buy_indices)
            self.sell_indices = np.array(cleaned_sell_indices)


class FixedStopStrategy(Strategy):
    def __init__(self, stop_loss_percent: float = 0, take_profit_percent: float = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_loss_percent = stop_loss_percent
        self.take_profit_percent = take_profit_percent
        self.stop = signals.FixedStopTake(stop_loss_percent=self.stop_loss_percent,
                                          take_profit_percent=self.take_profit_percent,
                                          interval=self.market_data.interval)

    def _apply_to_sub_df(self) -> NoReturn:
        """ Apply strategy to sub df, get stop indices """
        if (buy_indices := self._get_signal_indices('buy')) is not False:
            self.buy_indices, self.sell_indices = self.stop.get_signals(self.market_data.current_price,
                                                                        self.market_data.current_index,
                                                                        buy_indices)

    def _get_trades(self) -> NoReturn:
        """ Calculate sell prices with stop """
        if self.buy_indices.shape[0] != 0:
            buy_price = self._get_price_by_indices(self.buy_indices)
            sell_price = self._get_price_by_indices(self.sell_indices)
            trades = sell_price / buy_price - 1
            good_buy_index_indices = trades > 0
            bad_buy_index_indices = trades < 0

            good_buy_indices = self.buy_indices[good_buy_index_indices]
            bad_buy_indices = self.buy_indices[bad_buy_index_indices]

            good_prices = self._get_price_by_indices(good_buy_indices)
            bad_prices = self._get_price_by_indices(bad_buy_indices)

            take_prices = good_prices * (1 + self.take_profit_percent)
            stop_prices = bad_prices * (1 + self.stop_loss_percent)

            np.put(sell_price, np.argwhere(good_buy_index_indices), take_prices)
            np.put(sell_price, np.argwhere(bad_buy_index_indices), stop_prices)

            self.trades = sell_price / buy_price - 1
            self.real_sell_prices = sell_price
