from kaneshi.binance_data.binance_data import BinanceVisionData

s_date = (2021, 1, 1)
e_date = (2023, 4, 1)
interval = '1m'
type_ = 'monthly'
symbols = ['BTC']

symbols = [symbol + 'USDT' for symbol in symbols]

with BinanceVisionData(s_date=s_date,
                       e_date=e_date,
                       interval=interval,
                       symbols=symbols,
                       type_=type_) as binance_data:
    binance_data.get_candles()
