import telegram
from binance.client import Client
from binance import ThreadedWebsocketManager
from api import T_TOKEN, T_CHAT_ID, B_API_KEY, B_API_SECRET

from typing import NoReturn, Callable, Tuple, List, Union


class DefaultClient:
    def __init__(self, *args, **kwargs):
        self.symbol = None
        self.order_id = None
        self.base_asset_quantity = None

    @staticmethod
    def send_to_telegram(msg: str) -> NoReturn:
        """ Send message to telegram bot """
        bot = telegram.Bot(token=T_TOKEN)
        bot.sendMessage(chat_id=T_CHAT_ID, text=msg)

    def start_socket(self, message_handler: Callable) -> NoReturn:
        """ Start kline socket """
        twm = ThreadedWebsocketManager()
        twm.start()
        twm.start_kline_socket(callback=message_handler, symbol=self.symbol)
        twm.join()

    def update_price(self, *args, **kwargs):
        raise NotImplementedError

    def buy_market(self, *args, **kwargs):
        raise NotImplementedError

    def sell_market(self, *args, **kwargs):
        raise NotImplementedError

    def sell_stop(self, *args, **kwargs):
        raise NotImplementedError

    def check_order(self, *args, **kwargs):
        raise NotImplementedError

    def cancel_order(self, *args, **kwargs):
        raise NotImplementedError


# NOT TESTED
class BinanceClient(DefaultClient):
    def __init__(self,
                 base_asset: str,
                 quote_asset: str,
                 stop_shift: float = 1.001):
        super().__init__()
        self.stop_shift = stop_shift
        self.base_asset = base_asset
        self.quote_asset = quote_asset
        self.symbol = base_asset + quote_asset
        # self.client = Client(B_API_KEY, B_API_SECRET)
        self.client = Client()
        self.quantity_decimals, self.price_decimals = self.count_n_decimals(self.get_decimals())

    @staticmethod
    def round_down(value: float, n_decimals: int) -> float:
        """ Round up order quantity to minimum allowable value  """
        factor = 1 / (10 ** n_decimals)
        return (value // factor) * factor

    @staticmethod
    def count_n_decimals(values: Tuple[float, float]) -> List[int]:
        """ Count number of decimals to round up """
        rets = []
        for value in values:
            counter = 0
            while round(value) <= 0:
                value *= 10
                counter += 1
            rets.append(counter)
        return rets

    def get_decimals(self) -> Tuple[float, float]:
        """ Get symbol filters decimals to values """
        price_decimals, quantity_decimals = 0, 0
        symbol_info = self.client.get_symbol_info(self.symbol)
        filters = symbol_info['filters']
        for filter_ in filters:
            filter_type = list(filter_.values())[0]
            if filter_type == 'LOT_SIZE':
                quantity_decimals = float(filter_['stepSize'])
            if filter_type == 'PRICE_FILTER':
                price_decimals = float(filter_['tickSize'])
        return quantity_decimals, price_decimals

    def get_balance(self, asset: str) -> float:
        """ Get the specific asset current balance """
        balance = self.client.get_asset_balance(asset)
        return float(balance['free'])

    def check_balance(self, quantity: float, kind: str) -> bool:
        """ Check whether current balance sufficient to place an order """
        balance = self.get_balance(getattr(self, f'{kind}_asset'))
        if quantity <= balance:
            return True
        raise Exception('Insufficient balance')

    def buy_market(self, quote_order_quantity: int) -> Union[dict, bool]:
        """ Place buy market order """
        if self.check_balance(quote_order_quantity, 'quote'):
            order = self.client.create_order(
                side='BUY',
                type='MARKET',
                symbol=self.symbol,
                quoteOrderQty=quote_order_quantity,
                newOrderRespType='FULL')
            if order['status'] == 'FILLED':
                self.base_asset_quantity = float(order['cummulativeQuoteQty'])
                return order
            return False

    def sell_market(self) -> Union[dict, bool]:
        """ Place sell market order """
        quantity = self.round_down(self.base_asset_quantity, self.quantity_decimals)
        if self.check_balance(quantity, 'base'):
            order = self.client.create_order(
                side='SELL',
                type='MARKET',
                symbol=self.symbol,
                quantity=quantity,
                newOrderRespType='FULL')
            if order['status'] == 'FILLED':
                self.base_asset_quantity = None
                return order
            return False

    def sell_stop(self, stop_price: float) -> Union[dict, bool]:
        """ Place stop loss limit order """
        quantity = self.round_down(self.base_asset_quantity, self.quantity_decimals)
        if self.check_balance(quantity, 'base'):
            order = self.client.create_order(
                side='SELL',
                symbol=self.symbol,
                type='STOP_LOSS_LIMIT',
                quantity=quantity,
                stopPrice=round(stop_price * self.stop_shift, self.price_decimals),
                price=round(stop_price, self.price_decimals),
                timeInForce='GTC',
                newOrderRespType='FULL')
            if order['status'] == 'NEW':
                self.order_id = order['orderId']
                return order
            return False

    def check_order(self) -> Union[dict, bool]:
        """ Get last placed stop loss limit order info """
        if self.order_id is not None:
            order_info = self.client.get_order(symbol=self.symbol, orderId=self.order_id)
            return order_info
        return False

    def cancel_order(self) -> Union[dict, bool]:
        """ Cancel current stop loss limit order """
        order_info = self.client.cancel_order(symbol=self.symbol, orderId=self.order_id)
        if order_info['status'] == 'CANCELED':
            self.order_id = None
            return order_info
        return False


class EmptyClient(BinanceClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0
        self.current_price = None
        self.current_stop_price = None
        self.sell_stop_order = None

    def update_price(self, current_price: float) -> NoReturn:
        """ Update current price value """
        self.current_price = current_price

    def buy_market(self, quantity: float) -> dict:
        """ Simulate buy market order return """
        quote_quantity = quantity / self.current_price
        order = {'symbol': self.symbol,
                 'orderId': self.counter,
                 'orderListId': -1,
                 'clientOrderId': '1',
                 'transactTime': 1,
                 'price': '0.00000000',
                 'origQty': str(quantity),
                 'executedQty': str(quantity),
                 'cummulativeQuoteQty': str(quote_quantity),
                 'status': 'FILLED',
                 'timeInForce': 'GTC',
                 'type': 'MARKET',
                 'side': 'BUY',
                 'workingTime': 1,
                 'fills': [{'price': str(self.current_price),
                            'qty': str(quantity),
                            'commission': '0.00000000',
                            'commissionAsset': 'BNB',
                            'tradeId': 1}],
                 'selfTradePreventionMode': 'NONE'}
        self.counter += 1
        self.base_asset_quantity = float(order['cummulativeQuoteQty'])
        return order

    def sell_market(self) -> dict:
        """ Simulate sell market order return """
        quantity = self.round_down(self.base_asset_quantity, self.quantity_decimals)
        quote_quantity = quantity * self.current_price
        order = {'symbol': self.symbol,
                 'orderId': self.counter,
                 'orderListId': -1,
                 'clientOrderId': '1',
                 'transactTime': 1,
                 'price': '0.00000000',
                 'origQty': str(quantity),
                 'executedQty': str(quantity),
                 'cummulativeQuoteQty': str(quote_quantity),
                 'status': 'FILLED',
                 'timeInForce': 'GTC',
                 'type': 'MARKET',
                 'side': 'SELL',
                 'workingTime': 1,
                 'fills': [{'price': str(self.current_price),
                            'qty': str(quantity),
                            'commission': '0.00000000',
                            'commissionAsset': 'BNB',
                            'tradeId': 1}],
                 'selfTradePreventionMode': 'NONE'}
        self.counter += 1
        self.base_asset_quantity = None
        return order

    def sell_stop(self, stop_price: float) -> dict:
        """ Simulate sell stop limit order return """
        self.current_stop_price = stop_price
        quantity = self.round_down(self.base_asset_quantity, self.quantity_decimals)
        order = {'symbol': self.symbol,
                 'orderId': self.counter,
                 'orderListId': -1,
                 'clientOrderId': 'ZxmGgT7INPsB663Xj1Q5c9',
                 'transactTime': 1,
                 'price': (round(stop_price, self.price_decimals)),
                 'origQty': str(quantity),
                 'executedQty': '0.00000000',
                 'cummulativeQuoteQty': '0.00000000',
                 'status': 'NEW',
                 'timeInForce': 'GTC',
                 'type': 'STOP_LOSS_LIMIT',
                 'side': 'SELL',
                 'stopPrice': str(round(stop_price * self.stop_shift, self.price_decimals)),
                 'workingTime': -1,
                 'fills': [],
                 'selfTradePreventionMode': 'NONE'}
        self.order_id = self.counter
        self.counter += 1
        self.sell_stop_order = order
        return order

    def cancel_order(self) -> dict:
        """ Simulate sell stop limit canceling return """
        order = {'symbol': self.symbol,
                 'origClientOrderId': '1',
                 'orderId': self.sell_stop_order['orderId'],
                 'orderListId': -1,
                 'clientOrderId': '1',
                 'price': self.sell_stop_order['price'],
                 'origQty': self.sell_stop_order['origQty'],
                 'executedQty': '0.00000000',
                 'cummulativeQuoteQty': '0.00000000',
                 'status': 'CANCELED',
                 'timeInForce': 'GTC',
                 'type': 'STOP_LOSS_LIMIT',
                 'side': 'SELL',
                 'stopPrice': self.sell_stop_order['stopPrice'],
                 'selfTradePreventionMode': 'NONE'}
        self.order_id = None
        return order

    def check_order(self) -> Union[dict, bool]:
        """ Simulate sell stop limit order get info return """
        status = 'NEW' if self.current_price > self.current_stop_price else 'FILLED'
        if self.order_id is not None:
            order = {'symbol': self.symbol,
                     'orderId': self.sell_stop_order['orderId'],
                     'orderListId': -1,
                     'clientOrderId': '1',
                     'price': self.sell_stop_order['price'],
                     'origQty': self.sell_stop_order['origQty'],
                     'executedQty': '0.00000000',
                     'cummulativeQuoteQty': '0.00000000',
                     'status': status,
                     'timeInForce': 'GTC',
                     'type': 'STOP_LOSS_LIMIT',
                     'side': 'SELL',
                     'stopPrice': self.sell_stop_order['stopPrice'],
                     'icebergQty': '0.00000000',
                     'time': 1,
                     'updateTime': 1,
                     'isWorking': False,
                     'workingTime': -1,
                     'origQuoteOrderQty': '0.00000000',
                     'selfTradePreventionMode': 'NONE'}
            return order
        return False
