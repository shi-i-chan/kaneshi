# kaneshi

<details>
<summary>
Getting binance data
</summary>
  
```python
from kaneshi.binance_data.binance_data import BinanceVisionData

config = {
    's_date': (2021, 1, 1),
    'e_date': (2023, 4, 1),
    'type_':  'monthly',
    'symbols': ['BTCUSDT', 'ETHUSDT', 'XRPUSDT']
}

with BinanceVisionData(**config) as binance_data:
    binance_data.get_candles()
```
Files `raw_BTCUSDT_1m.h5`, `raw_ETHUSDT_1m.h5`, etc will appera in folder `kaneshi\kaneshi\data\market_data`
  
</details>

