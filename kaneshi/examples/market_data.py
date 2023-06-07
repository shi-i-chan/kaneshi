from kaneshi.core.market_data import MarketData

config = {'symbol': 'BTCUSDT',
          'interval': 1,
          's_date': (2021, 1, 1),
          'e_date': (2023, 1, 1),
          'columns': 'Close',
          'price_type': 'Close',
          }

market = MarketData.from_config(**config)

print(market.raw_df.head().to_string())
print(market.raw_df.tail().to_string())
