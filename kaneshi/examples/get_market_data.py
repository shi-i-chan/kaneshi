from kaneshi.utils.binance_data import BinanceVisionData

s_date = (2021, 1, 1)
e_date = (2023, 4, 1)
interval = '1d'
type_ = 'monthly'
symbols = ['BTC', 'ETH', 'XRP', 'MATIC']

symbols = [symbol + 'USDT' for symbol in symbols]

with BinanceVisionData(start_date=s_date,
                       end_date=e_date,
                       interval=interval,
                       symbols=symbols,
                       type_=type_) as binance_data:
    binance_data.get_candles()
