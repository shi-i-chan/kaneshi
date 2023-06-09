import numpy as np

from kaneshi.core import signals, indicators
from kaneshi.core.def_strategy import Strategy, ClearStrategy, FixedStopStrategy

from numpy.typing import NDArray


class SMACStrategy(Strategy):
    def __init__(self,
                 sma: int = None,
                 lma: int = None,
                 *args, **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self.strategy_label = 'smac'
        self.sma = int(sma)
        self.lma = int(lma)

        self.price_ind = indicators.Price()
        self.sma_ind = indicators.Simple_MA(self.sma, color='orange')
        self.lma_ind = indicators.Simple_MA(self.lma, color='green')
        self.cross_down_signal = signals.Crossover(kind='down', minor_indicator=self.sma_ind,
                                                   major_indicator=self.lma_ind)
        self.cross_up_signal = signals.Crossover(kind='up', minor_indicator=self.sma_ind,
                                                 major_indicator=self.lma_ind)

        self.buy_conditions = {'signal_generators': [self.cross_up_signal], 'merge_type': 'and'}
        self.sell_conditions = {'signal_generators': [self.cross_down_signal], 'merge_type': 'and'}
        self.plot_config = [{'major': self.price_ind, 'minor': [self.sma_ind, self.lma_ind]}]

    def _fix_bug_with_first_signals(self, raw_indices: NDArray[np.int32]) -> NDArray[np.int32]:
        """ Remove signal if it arose due to nan """
        bad_indices_indices = np.argwhere((raw_indices == self.sma - 1) |
                                          (raw_indices == self.lma - 1))  # NOQA
        raw_indices = np.delete(raw_indices, bad_indices_indices)
        return raw_indices


class SMACClear(SMACStrategy, ClearStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class SMACFixedStop(SMACStrategy, FixedStopStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._add_stops_to_plot()


class RSIStrategy(Strategy):
    def __init__(self,
                 rsi_period: int = None,
                 bottom_edge: int = None,
                 upper_edge: int = None,
                 *args, **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self.strategy_label = 'rsi'
        self.rsi_period = rsi_period
        self.bottom_edge = bottom_edge
        self.upper_edge = upper_edge

        self.price_ind = indicators.Price()
        self.rsi_ind = indicators.RSI(rsi_period)
        self.bottom_edge_ind = indicators.Constant(bottom_edge, color='green',
                                                   label=f'Down edge {self.bottom_edge}')
        self.upper_edge_ind = indicators.Constant(upper_edge, color='red',
                                                  label=f'Up edge {self.upper_edge}')
        self.cross_down_signal = signals.Crossover('up', self.rsi_ind, self.bottom_edge_ind)
        self.cross_up_signal = signals.Crossover('up', self.rsi_ind, self.upper_edge_ind)
        self.buy_conditions = {'signal_generators': [self.cross_down_signal]}
        self.sell_conditions = {'signal_generators': [self.cross_up_signal]}
        self.plot_config = [{'major': self.price_ind},
                            {'major': self.rsi_ind, 'minor': [self.bottom_edge_ind, self.upper_edge_ind]}]

    def _fix_bug_with_first_signals(self, raw_indices: NDArray[np.int32]) -> NDArray[np.int32]:
        """ Remove signal if it arose due to nan """
        bad_indices_indices = np.argwhere((raw_indices == self.rsi_period))  # NOQA
        raw_indices = np.delete(raw_indices, bad_indices_indices)
        return raw_indices


class RSIClear(RSIStrategy, ClearStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RSIFixedStop(RSIStrategy, FixedStopStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._add_stops_to_plot()
