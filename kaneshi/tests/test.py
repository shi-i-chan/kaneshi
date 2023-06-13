import numpy as np
from unittest import TestCase
import matplotlib.pyplot as plt

from kaneshi.utils import read_df
from kaneshi.core.market_data import MarketData

from kaneshi.core.strategies import RSIOBSClear, RSIOBSFixedStop, SMACClear, SMACFixedStop, RSIOBSOneEdgeClear, \
    RSIOBSOneEdgeFixedStop, PCTFixedStop, BBClear, BBFixedStop

from numpy.typing import NDArray

btc_2021 = read_df('BTC_2021')
market_data = MarketData(raw_df=btc_2021, interval=60)


def check_plots(real_image: NDArray, plot_fn: str) -> bool:
    expected_image = plt.imread('charts/' + plot_fn)
    return np.array_equal(real_image, expected_image)


class TestRSIOBSStrategies(TestCase):
    def setUp(self) -> None:
        self.s_name = 'RSIOBS'
        self.params = {'rsi_period': 7,
                       'bottom_edge': 20,
                       'upper_edge': 80}

        self.stop_loss_percent = -0.01
        self.take_profit_percent = 0.01

        self.clear = RSIOBSClear(**self.params,
                                 market_data=market_data).apply()

        self.stop_loss = RSIOBSFixedStop(**self.params,
                                         stop_loss_percent=self.stop_loss_percent,
                                         market_data=market_data).apply()

        self.take_profit = RSIOBSFixedStop(**self.params,
                                           take_profit_percent=self.take_profit_percent,
                                           market_data=market_data).apply()

        self.stop_take = RSIOBSFixedStop(**self.params,
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


class TestRSIOBSOneEdgeStrategies(TestCase):
    def setUp(self) -> None:
        self.s_name = 'RSIOBSOneEdge'
        self.params = {'rsi_period': 7,
                       'edge': 30}

        self.stop_loss_percent = -0.01
        self.take_profit_percent = 0.01

        self.clear = RSIOBSOneEdgeClear(**self.params,
                                        market_data=market_data).apply()

        self.stop_loss = RSIOBSOneEdgeFixedStop(**self.params,
                                                stop_loss_percent=self.stop_loss_percent,
                                                market_data=market_data).apply()

        self.take_profit = RSIOBSOneEdgeFixedStop(**self.params,
                                                  take_profit_percent=self.take_profit_percent,
                                                  market_data=market_data).apply()

        self.stop_take = RSIOBSOneEdgeFixedStop(**self.params,
                                                stop_loss_percent=self.stop_loss_percent,
                                                take_profit_percent=self.take_profit_percent,
                                                market_data=market_data).apply()

        self.strategies = {'clear': self.clear,
                           'stop_loss': self.stop_loss,
                           'take_profit': self.take_profit,
                           'stop_take': self.stop_take}

        self.reports = {'clear': [348, -7.206956, -28.564406, 0.270115, 1293.62069, 0.250511, -0.074707],
                        'stop_loss': [51, -58.57066, -44.517128, 0.0, 4256.470588, -0.01, -0.01],
                        'take_profit': [56, 47.519181, 60.510153, 1.0, 2401.071429, 0.01, 0.01],
                        'stop_take': [307, -15.079214, -15.296495, 0.550489, 279.674267, 0.01, -0.01]}

    def test_reports(self):
        for s_type, s in self.strategies.items():
            with self.subTest():
                self.assertEqual(s.generate_report(), self.reports[s_type])

    def test_plots(self):
        for s_type, s in self.strategies.items():
            with self.subTest(msg=f'{self.s_name}_{s_type}_plot'):
                self.assertTrue(check_plots(s.plot(plot_type='array'), f'{self.s_name}_{s_type}_plot.png'))


class TestSMACStrategies(TestCase):
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


class TestPCTStrategies(TestCase):
    def setUp(self) -> None:
        self.s_name = 'PCT'
        self.params = {'pct_period': 14,
                       'pct_edge': 0.01}
        self.stop_loss = PCTFixedStop(**self.params,
                                      stop_loss_percent=-0.01,
                                      market_data=market_data).apply()

        self.take_profit = PCTFixedStop(**self.params,
                                        take_profit_percent=0.01,
                                        market_data=market_data).apply()

        self.stop_take = PCTFixedStop(**self.params,
                                      stop_loss_percent=-0.01,
                                      take_profit_percent=0.01,
                                      market_data=market_data).apply()

        self.strategies = {'stop_loss': self.stop_loss,
                           'take_profit': self.take_profit,
                           'stop_take': self.stop_take}

        self.reports = {'stop_loss': [61, -70.055103, -50.569612, 0.0, 2948.852459, -0.01, -0.01],
                        'take_profit': [67, 56.853306, 76.144494, 1.0, 2846.865672, 0.01, 0.01],
                        'stop_take': [411, -72.610387, -52.63497, 0.486618, 280.729927, 0.01, -0.01]}

    def test_reports(self):
        for s_type, s in self.strategies.items():
            with self.subTest():
                self.assertEqual(s.generate_report(), self.reports[s_type])

    def test_plots(self):
        for s_type, s in self.strategies.items():
            with self.subTest(msg=f'{s_type}_plot'):
                self.assertTrue(check_plots(s.plot(plot_type='array'), f'{self.s_name}_{s_type}_plot.png'))


class TestBBStrategies(TestCase):
    def setUp(self) -> None:
        self.s_name = 'BB'
        self.params = {'ma_period': 150}

        self.clear = BBClear(**self.params, market_data=market_data).apply()

        self.stop_loss = BBFixedStop(**self.params,
                                     stop_loss_percent=-0.01,
                                     market_data=market_data).apply()

        self.take_profit = BBFixedStop(**self.params,
                                       take_profit_percent=0.01,
                                       market_data=market_data).apply()

        self.stop_take = BBFixedStop(**self.params,
                                     stop_loss_percent=-0.01,
                                     take_profit_percent=0.01,
                                     market_data=market_data).apply()

        self.strategies = {'clear': self.clear,
                           'stop_loss': self.stop_loss,
                           'take_profit': self.take_profit,
                           'stop_take': self.stop_take}

        self.reports = {'clear': [17, 28.682762, 18.316673, 0.823529, 13722.352941, 0.182919, -0.332766],
                        'stop_loss': [55, -63.164437, -47.022316, 0.0, 4635.272727, -0.01, -0.01],
                        'take_profit': [40, 33.942272, 40.21275, 1.0, 679.5, 0.01, 0.01],
                        'stop_take': [76, -1.410719, -1.768144, 0.565789, 263.684211, 0.01, -0.01]}

    def test_reports(self):
        for s_type, s in self.strategies.items():
            with self.subTest():
                self.assertEqual(s.generate_report(), self.reports[s_type])

    def test_plots(self):
        for s_type, s in self.strategies.items():
            with self.subTest(msg=f'{s_type}_plot'):
                self.assertTrue(check_plots(s.plot(plot_type='array'), f'{self.s_name}_{s_type}_plot.png'))
