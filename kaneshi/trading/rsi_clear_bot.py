from kaneshi.trading.bots import ClearRSIBot
from kaneshi.trading.database import DataBase
from kaneshi.trading.clients import EmptyClient


clear_rsi_config = {'base_asset': 'XRP',
                    'quote_asset': 'USDT',
                    'rsi_period': 30,
                    'bottom_edge': 50,
                    'upper_edge': 55,
                    'quantity': 10,
                    'client': EmptyClient,
                    'db': DataBase,
                    'is_testing': True}

clear_rsi = ClearRSIBot(**clear_rsi_config)

clear_rsi.start_bot()
