# kaneshi

A little project with backtesting some algorithmic trading strategies, Binance trading bots, and some machine learning.

<details>
<summary>
Getting Binance data
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

<details open>
<summary>
Clear MA crossover strategy
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

![image](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/screens/smac_clear_report.png)

```python
s.plot(plot_type='plt')  
```
  
![image](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/screens/smac_clear_plot.png)
  
</details>

<details>
<summary>
Fixed stop MA crossover strategy
</summary>

[Example](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/examples/smac_det_stop_strategy.ipynb)
  
```python
from kaneshi.core.strategies import SMACFixedStop
  
s_config = {
    'sma': 14,
    'lma': 50,
    'stop_loss_percent': -0.01,
    'take_profit_percent': 0.01,
    'market_data': market,
}

s = SMACFixedStop(**s_config).apply()
  
s.generate_report()
```

![image](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/screens/smac_fixed_stop_report.png)
 
```python
s.plot(plot_type='plt')  
```
  
![image](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/screens/smac_fixed_stop_plot.png)
  
</details>

<details>
<summary>
Clear RSI strategy
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
  
![image](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/screens/rsi_clear_report.png)

```python
s.plot(plot_type='plt')
```

![image](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/screens/rsi_clear_plot.png)
  
</details>


<details open>
<summary>
Fixed stop RSI strategy
</summary>
  
[Example](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/examples/rsi_det_stop_strategy.ipynb)
 
```python
from kaneshi.core.strategies import RSIFixedStop
  
s_config = {
    'rsi_period': 14,
    'bottom_edge': 20,
    'upper_edge': 60,
    'stop_loss_percent': -0.01,
    'take_profit_percent': 0.01,
    'market_data': market,
}

s = RSIFixedStop(**s_config).apply()
  
s.generate_report()
```

![image](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/screens/rsi_fixed_stop_report.png)

```python
s.plot(plot_type='plt')  
```
  
![image](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/screens/rsi_fixed_stop_plot.png)
  
</details>

<details open>
<summary>
Fixed stop PCT strategy (not tested normally)
</summary>
  
[Example](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/examples/pct_fixed_stop_strategy.ipynb)
 
```python
from kaneshi.core.strategies import PCTFixedStop
  
s_config = {
    'pct_period': 14,
    'pct_edge': 0.01,
    'stop_loss_percent': -0.01,
    'take_profit_percent': 0.01,
    'market_data': market,
}

s = PCTFixedStop(**s_config).apply()
  
s.generate_report()
```

![image](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/screens/pct_fixed_stop_report.png)

```python
s.plot(plot_type='plt')  
```
  
![image](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/screens/pct_fixed_stop_plot.png)
  
</details>

<details>
<summary>
Create dataset
</summary>
  
[Example](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/examples/create_dataset.ipynb)
 
```python
# init and apply strategy
  
s.create_dataset(lookback=100, dataset_fn='file_name')
```

File `file_name.h5` will appear in folder `kaneshi/kaneshi/data/datasets`.

There is `n` training examples, where `n` equal to the strategy number of trades. `y_data` is trades labels (1 or 0 if trade profitable or unprofitable). `x_data` is `n=lookback` candles before specific trade (Actually, `n` rows in market_data dataframe with corresponding columns).

![image](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/screens/dataset_plot.png)
  
</details>

<details>
<summary>
Dataset preparation
</summary>
  
[Example](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/examples/datasets.ipynb)
 
Init, split, normalize, one-hot encoding, showing, etc.

Dataset classes plot example.

The examples are white noise, so the models are unlikely to show any results.
  

![image](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/screens/dataset_classes.png)
  
</details>


<details>
<summary>
CNN 1D model example
</summary>
  
[Example](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/examples/cnn1d_model.ipynb)
 
As expected from the existing dataset, the model overfits and does not show any results. On the validation dataset the loss increased, not the accuracy.

Similar results are obtained with another architectures, tuning and other changes. The problem is data.
  
![image](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/screens/cnn_1d_model.png)
  
</details>


<details open>
<summary>
Clear RSI strategy Binance trading bot (not tested normally)
</summary>
  
[Example](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/trading/rsi_clear_bot.py)

  
```python
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
```

[Bot results visualization](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/examples/rsi_bot_visual.ipynb)

![image](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/screens/rsi_bot_visual.png)
  
[Bot realtime dashboard (draft)](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/examples/rsi_bot_dashboard.ipynb)

![image](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/screens/rsi_clear_bot_dashboard.png)
  
</details>
