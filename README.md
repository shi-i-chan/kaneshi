# kaneshi

<details>
<summary>
Getting binance data
</summary>

[Example](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/examples/get_binance_data.ipynb)
  
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

Files `raw_BTCUSDT_1m.h5`, `raw_ETHUSDT_1m.h5`, etc will appear in folder `kaneshi\kaneshi\data\market_data`
  
</details>

<details>
<summary>
Read market data
</summary>

[Example](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/examples/market_data.ipynb)

```python
from kaneshi.core.market_data import MarketData
  
config = {'symbol': 'BTCUSDT',
          'interval': 1,
          's_date': (2021, 1, 1),
          'e_date': (2021, 2, 1),
          'columns': 'Close',
          'price_type': 'Close',
          }

market = MarketData.from_config(**config)
```

</details>

<details>
<summary>
Clear moving average strategy
</summary>

[Example](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/examples/smac_clear_strategy.ipynb)
  
```python
from kaneshi.core.strategies import SMACClear

s_config = {
    'sma': 14,
    'lma': 50,
    'market_data': market,
}

s = SMACClear(**s_config).apply()

s.generate_report()

```

![image](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/screens/smac_clear_generate_report.png)

```python
s.plot(plot_type='plt')  
```
  
![image](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/screens/smac_clear_plot.png)
  
</details>

<details>
<summary>
Moving average strategy with fixed stops
</summary>

[Example](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/examples/smac_det_stop_strategy.ipynb)
  
```python
from kaneshi.core.strategies import SMACDetStop
  
s_config = {
    'sma': 14,
    'lma': 50,
    'stop_loss_percent': -0.01,
    'take_profit_percent': 0.01,
    'market_data': market,
}

s = SMACDetStop(**s_config).apply()
  
s.generate_report()
```

![image](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/screens/smac_det_stop_generate_report.png)
 
```python
s.plot(plot_type='plt')  
```
  
![image](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/screens/smac_det_stop_plot.png)
  
</details>

<details>
<summary>
RSI clear strategy
</summary>
  
[Example](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/examples/rsi_clear_strategy.ipynb)
  
```python
from kaneshi.core.strategies import RSIClear
  
s_config = {
    'rsi_period': 14,
    'bottom_edge': 20,
    'upper_edge': 60,
    'market_data': market,
}

s = RSIClear(**s_config).apply()
 
s.generate_report()
```
  
![image](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/screens/rsi_clear_generate_report.png)

```python
s.plot(plot_type='plt')  
```

![image](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/screens/rsi_clear_plot.png)
  
</details>


<details>
<summary>
RSI strategy with fixed stops
</summary>
  
[Example](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/examples/rsi_det_stop_strategy.ipynb)
 
```python
from kaneshi.core.strategies import RSIDetStop
  
s_config = {
    'rsi_period': 14,
    'bottom_edge': 20,
    'upper_edge': 60,
    'stop_loss_percent': -0.01,
    'take_profit_percent': 0.01,
    'market_data': market,
}

s = RSIDetStop(**s_config).apply()
  
s.generate_report()
```

![image](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/screens/rsi_det_stop_generate_report.png)

```python
s.plot(plot_type='plt')  
```
  
![image](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/screens/rsi_det_stop_plot.png)
  
</details>
