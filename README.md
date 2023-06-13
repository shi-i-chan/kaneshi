# kaneshi

A little project with backtesting some algorithmic trading strategies, Binance trading bots, and some machine learning.

- ### Getting Binance data

<ul>

[Example](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/examples/get_binance_data.ipynb)

</ul>
  
- ### Read market data

<ul>

[Example](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/examples/market_data.ipynb)

</ul>

- ### Clear MA crossover strategy

<ul>

[Example](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/examples/smac_clear_strategy.ipynb)

</ul>

![image](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/tests/charts/SMAC_clear_plot.png)


- ### Fixed stop MA crossover strategy

<ul>

[Example](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/examples/smac_fixed_stop_strategy.ipynb)

</ul>

![image](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/tests/charts/SMAC_stop_take_plot.png)
 

- ### Clear RSI oversold overbought strategy
 
<ul>
 
[Example](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/examples/rsi_overbought_oversold_clear_strategy.ipynb)

</ul>
  
![image](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/tests/charts/RSIOBS_clear_plot.png)


- ### Fixed stop RSI oversold overbought strategy

<ul>

[Example](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/examples/rsi_overbought_oversold_fixed_stop_strategy.ipynb)
  
</ul>

![image](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/tests/charts/RSIOBS_stop_take_plot.png)

- ### Clear RSI oversold overbought one edge strategy
 
<ul>
 
[Example](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/examples/rsi_overbought_oversold_one_edge_clear_strategy.ipynb)

</ul>
  
![image](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/tests/charts/RSIOBSOneEdge_clear_plot.png)


- ### Fixed stop RSI oversold overbought one edge strategy

<ul>

[Example](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/examples/rsi_overbought_oversold_one_edge_fixed_stop_strategy.ipynb)
  
</ul>

![image](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/tests/charts/RSIOBSOneEdge_stop_take_plot.png)

- ### Fixed stop PCT strategy (not tested normally)

<ul>

[Example](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/examples/pct_fixed_stop_strategy.ipynb)

</ul>

![image](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/tests/charts/PCT_stop_take_plot.png)


- ### Clear Bollinger bands strategy (not tested normally)

<ul>

[Example](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/examples/bb_clear_strategy.ipynb)

</ul>

![image](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/tests/charts/BB_clear_plot.png)


- ### Fixed stop Bollinger bands strategy (not tested normally)

<ul>

[Example](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/examples/bb_fixed_stop_strategy.ipynb)

</ul>

![image](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/tests/charts/BB_stop_take_plot.png)

<details>
<summary>
Create dataset
</summary>
  
[Example](https://github.com/shi-i-chan/kaneshi/blob/main/kaneshi/examples/create_dataset.ipynb)

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


<details>
<summary>
Clear RSI oversold overbought strategy Binance trading bot (not tested normally)
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
