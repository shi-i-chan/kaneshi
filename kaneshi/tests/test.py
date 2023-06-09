import numpy as np
from unittest import TestCase
import matplotlib.pyplot as plt

from kaneshi.utils import read_df
from kaneshi.core.market_data import MarketData

from kaneshi.core.strategies import RSIClear, RSIFixedStop, SMACClear, SMACFixedStop

from numpy.typing import NDArray

btc_2021 = read_df('BTC_2021')
market_data = MarketData(raw_df=btc_2021, interval=60)


def check_plots(real_image: NDArray, plot_fn: str) -> bool:
    expected_image = plt.imread('charts/' + plot_fn)
    return np.array_equal(real_image, expected_image)


class TestClearRSIStrategy(TestCase):
    def setUp(self) -> None:
        self.s_name = 'RSI'
        self.params = {'rsi_period': 7,
                       'bottom_edge': 20,
                       'upper_edge': 80}

        self.stop_loss_percent = -0.01
        self.take_profit_percent = 0.01

        self.clear = RSIClear(**self.params,
                              market_data=market_data).apply()

        self.stop_loss = RSIFixedStop(**self.params,
                                      stop_loss_percent=self.stop_loss_percent,
                                      market_data=market_data).apply()

        self.take_profit = RSIFixedStop(**self.params,
                                        take_profit_percent=self.take_profit_percent,
                                        market_data=market_data).apply()

        self.stop_take = RSIFixedStop(**self.params,
                                      stop_loss_percent=self.stop_loss_percent,
                                      take_profit_percent=self.take_profit_percent,
                                      market_data=market_data).apply()

        self.strategies = {'clear': self.clear,
                           'stop_loss': self.stop_loss,
                           'take_profit': self.take_profit,
                           'stop_take': self.stop_take}

        self.reports = {'clear': [40, 13.903871, -0.596502, 0.675, 5694.0, 0.149937, -0.332206],
                        'stop_loss': [38, -43.640884, -35.527677, 0.0, 4967.368421, -0.01, -0.01],
                        'take_profit': [39, 33.093716, 39.032976, 1.0, 878.461538, 0.01, 0.01],
                        'stop_take': [130, -13.501684, -13.200688, 0.523077, 285.230769, 0.01, -0.01]}

    def test_reports(self):
        for s_type, s in self.strategies.items():
            with self.subTest():
                self.assertEqual(s.generate_report(), self.reports[s_type])

    def test_plots(self):
        for s_type, s in self.strategies.items():
            with self.subTest(msg=f'{self.s_name}_{s_type}_plot'):
                self.assertTrue(check_plots(s.plot(plot_type='array'), f'{self.s_name}_{s_type}_plot.png'))


class TestClearSMACrossover(TestCase):
    def setUp(self) -> None:
        self.s_name = 'SMAC'
        self.params = {'sma': 100,
                       'lma': 300}
        self.clear = SMACClear(**self.params, market_data=market_data).apply()

        self.stop_loss = SMACFixedStop(**self.params,
                                       stop_loss_percent=-0.01,
                                       market_data=market_data).apply()

        self.take_profit = SMACFixedStop(**self.params,
                                         take_profit_percent=0.01,
                                         market_data=market_data).apply()

        self.stop_take = SMACFixedStop(**self.params,
                                       stop_loss_percent=-0.01,
                                       take_profit_percent=0.01,
                                       market_data=market_data).apply()

        self.strategies = {'clear': self.clear,
                           'stop_loss': self.stop_loss,
                           'take_profit': self.take_profit,
                           'stop_take': self.stop_take}

        self.reports = {'clear': [12, -50.433876, -42.026931, 0.083333, 9950.0, 0.184575, -0.11083],
                        'stop_loss': [14, -16.07822, -14.931433, 0.0, 2190.0, -0.01, -0.01],
                        'take_profit': [12, 10.182682, 10.671616, 1.0, 1830.0, 0.01, 0.01],
                        'stop_take': [14, -2.099213, -2.147399, 0.5, 351.428571, 0.01, -0.01]}

    def test_reports(self):
        for s_type, s in self.strategies.items():
            with self.subTest():
                self.assertEqual(s.generate_report(), self.reports[s_type])

    def test_plots(self):
        for s_type, s in self.strategies.items():
            with self.subTest(msg=f'{s_type}_plot'):
                self.assertTrue(check_plots(s.plot(plot_type='array'), f'{self.s_name}_{s_type}_plot.png'))
