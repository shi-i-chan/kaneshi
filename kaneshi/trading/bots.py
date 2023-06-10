import numpy as np
import pandas as pd
from datetime import datetime

from kaneshi.core.indicators import RSI
from kaneshi.trading.database import DataBase
from kaneshi.trading.clients import DefaultClient

from typing import Type, NoReturn

DefaultClient = Type[DefaultClient]
DataBase = Type[DataBase]


class RSIBotVariables:
    def __init__(self,
                 base_asset: str = None,
                 quote_asset: str = None,
                 rsi_period: int = None,
                 bottom_edge: int = None,
                 upper_edge: int = None,
                 quantity: float = None,
                 db: DataBase = None,
                 client: DefaultClient = None,
                 is_testing: bool = False,
                 ):
        self.base_asset = base_asset
        self.quote_asset = quote_asset
        self.RSI_PERIOD = rsi_period
        self.BOTTOM_EDGE = bottom_edge
        self.UPPER_EDGE = upper_edge
        self.quantity = quantity
        self.db = db
        self.is_testing = is_testing
        self.client = client(self.base_asset, self.quote_asset)

        self.price_list = []
        self.in_position = False
        self.current_index = None
        self.last_rsi, self.current_rsi = np.nan, np.nan
        self.rsi_indicator = RSI(self.RSI_PERIOD)

    def calc_rsi(self) -> NoReturn:
        """ Calculate and save current RSI value """
        current_rsi = self.rsi_indicator.apply_to(np.array(self.price_list))[-1]
        self.last_rsi, self.current_rsi = self.current_rsi, current_rsi

    def message_handler(self, message: dict) -> NoReturn:
        """ Handle socket message with candles data """
        try:
            kline = message['k']
            # if kline['x']:
            current_close = float(kline['c'])
            # current_date = datetime.fromtimestamp(int(kline['t']) / 1e3)
            current_date = datetime.fromtimestamp(int(message['E']) / 1e3)
            self.process_message(current_date, current_close)
        except Exception as e:
            print(e)

    def start_bot(self) -> NoReturn:
        """ Start bot """
        self.client.start_socket(self.message_handler)

    def process_message(self, *args, **kwargs):
        raise NotImplementedError


# NOT TESTED
class ClearRSIBot(RSIBotVariables):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label = 'clear_rsi_{0}_{1}_{2}_{3}'.format(self.base_asset + self.quote_asset,
                                                        self.RSI_PERIOD, self.BOTTOM_EDGE, self.UPPER_EDGE)
        self.db = self.db(self.label)
        self.current_buy_order = None
        self.current_sell_order = None

    def process_message(self, current_index: np.datetime64, current_price: float) -> NoReturn:
        """ Calc and save data, create orders """
        if self.is_testing:
            self.client.update_price(current_price)
        self.current_index = current_index
        self.price_list.append(current_price)
        if len(self.price_list) > self.RSI_PERIOD + 3:
            self.calc_rsi()
            if not self.in_position:
                self.__check_buy_signal()
            if self.in_position:
                self.__check_sell_signal()
        self.__save_data()

    def __check_buy_signal(self) -> NoReturn:
        """ Check whether the buy signal was triggered """
        if self.last_rsi < self.BOTTOM_EDGE < self.current_rsi:
            if order := self.client.buy_market(self.quantity):
                self.in_position = True
            self.current_buy_order = order

    def __check_sell_signal(self) -> NoReturn:
        """" Check whether the sell signal was triggered """
        if self.last_rsi < self.UPPER_EDGE < self.current_rsi:
            if order := self.client.sell_market():
                self.in_position = False
            self.current_sell_order = order

    def __save_data(self) -> NoReturn:
        """ Save current data to database and reset data variables """
        data = [self.current_index, self.price_list[-1], self.current_rsi,
                str(self.current_buy_order), str(self.current_sell_order)]
        df = pd.DataFrame([data], columns=['index', 'price', 'rsi', 'buy_order', 'sell_order'])
        self.db.save_data('data', df)
        self.current_index = None
        self.current_buy_order = None
        self.current_sell_order = None
