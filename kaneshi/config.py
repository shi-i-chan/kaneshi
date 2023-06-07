import os
import numpy as np
import pandas as pd

EMPTY_DT_ARRAY = np.array([], dtype='datetime64')
EMPTY_ARRAY = np.array([])
EMPTY_DF = pd.DataFrame()

DEF_DF_EXTENSION = 'h5'
DEF_ROUND = 6
DEF_COMMISSION = 0.00075

DEF_PLT_FIGSIZE = (14, 7)
DEF_PLOTLY_WIDTH = 1000
DEF_PLOTLY_HEIGHT = 500

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEF_MARKET_DATA_PATH = rf'{ROOT_DIR}\data\market_data'

DEF_REPORT_CONFIG = [
    'n_trades',
    'equal_return',
    'cumulative_return',
    'win_rate',
    'mean_trade_time',
    'max_return',
    'max_drawdown',
]


# from kaneshi.core.def_strategy import Strategy as def_strategy
# from typing import TypeVar
#
# Strategy = TypeVar('Strategy', bound=def_strategy)