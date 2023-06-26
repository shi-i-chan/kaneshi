# kaneshi

A little project with backtesting some algorithmic trading strategies, Binance trading bots, and some machine learning.

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/shichaan)

<a href="#tron" alt="TRON"><img src="https://img.shields.io/badge/Donate-TRC--20-7DFDFE?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA2MTAgNjEwIiB3aWR0aD0iMjUwMCIgaGVpZ2h0PSIyNTAwIj48Y2lyY2xlIGN4PSIzMDUiIGN5PSIzMDUiIHI9IjMwNSIvPjxwYXRoIGQ9Ik01MDUuNCAyMTQuN2MtMTcuMy0xMi4xLTM1LjgtMjUtNTMuOS0zNy44LS40LS4zLS44LS42LTEuMy0uOS0yLTEuNS00LjMtMy4xLTcuMS00bC0uMi0uMWMtNDguNC0xMS43LTk3LjYtMjMuNy0xNDUuMi0zNS4zLTQzLjItMTAuNS04Ni4zLTIxLTEyOS41LTMxLjUtMS4xLS4zLTIuMi0uNi0zLjQtLjktMy45LTEuMS04LjQtMi4zLTEzLjItMS43LTEuNC4yLTIuNi43LTMuNyAxLjRsLTEuMiAxYy0xLjkgMS44LTIuOSA0LjEtMy40IDUuNGwtLjMuOHY0LjZsLjIuN2MyNy4zIDc2LjUgNTUuMyAxNTQuMSA4Mi4zIDIyOS4yIDIwLjggNTcuOCA0Mi40IDExNy43IDYzLjUgMTc2LjUgMS4zIDQgNSA2LjYgOS42IDdoMWM0LjMgMCA4LjEtMi4xIDEwLTUuNWw3OS4yLTExNS41YzE5LjMtMjguMSAzOC42LTU2LjMgNTcuOS04NC40IDcuOS0xMS41IDE1LjgtMjMuMSAyMy43LTM0LjYgMTMtMTkgMjYuNC0zOC42IDM5LjctNTcuN2wuNy0xdi0xLjJjLjMtMy41LjQtMTAuNy01LjQtMTQuNW0tOTIuOCA0Mi4xYy0xOC42IDkuNy0zNy42IDE5LjctNTYuNyAyOS42IDExLjEtMTEuOSAyMi4zLTIzLjkgMzMuNC0zNS44IDEzLjktMTUgMjguNC0zMC41IDQyLjYtNDUuN2wuMy0uM2MxLjItMS42IDIuNy0zLjEgNC4zLTQuNyAxLjEtMS4xIDIuMy0yLjIgMy40LTMuNSA3LjQgNS4xIDE0LjkgMTAuMyAyMi4xIDE1LjQgNS4yIDMuNyAxMC41IDcuNCAxNS45IDExLjEtMjIgMTEuMi00NCAyMi43LTY1LjMgMzMuOW0tNDcuOC00LjhjLTE0LjMgMTUuNS0yOS4xIDMxLjQtNDMuOCA0Ny4xLTI4LjUtMzQuNi01Ny42LTY5LjctODUuOC0xMDMuNi0xMi44LTE1LjQtMjUuNy0zMC45LTM4LjUtNDYuM2wtLjEtLjFjLTIuOS0zLjMtNS43LTYuOS04LjUtMTAuMy0xLjgtMi4zLTMuNy00LjUtNS42LTYuOCAxMS42IDMgMjMuMyA1LjggMzQuOCA4LjUgMTAuMSAyLjQgMjAuNiA0LjkgMzAuOSA3LjUgNTggMTQuMSAxMTYuMSAyOC4yIDE3NC4xIDQyLjMtMTkuMyAyMC42LTM4LjcgNDEuNS01Ny41IDYxLjdtLTUwLjMgMTk0LjljMS4xLTEwLjUgMi4zLTIxLjMgMy4zLTMxLjkuOS04LjUgMS44LTE3LjIgMi43LTI1LjUgMS40LTEzLjMgMi45LTI3LjEgNC4xLTQwLjZsLjMtMi40YzEtOC42IDItMTcuNSAyLjYtMjYuNCAxLjEtLjYgMi4zLTEuMiAzLjYtMS43IDEuNS0uNyAzLTEuMyA0LjUtMi4yIDIzLjEtMTIuMSA0Ni4yLTI0LjIgNjkuNC0zNi4yIDIzLjEtMTIgNDYuOC0yNC40IDcwLjMtMzYuNy0yMS40IDMxLTQyLjkgNjIuMy02My43IDkyLjgtMTcuOSAyNi4xLTM2LjMgNTMtNTQuNiA3OS41LTcuMiAxMC42LTE0LjcgMjEuNC0yMS44IDMxLjgtOCAxMS42LTE2LjIgMjMuNS0yNC4yIDM1LjQgMS0xMiAyLjItMjQuMSAzLjUtMzUuOU0xNzUuMSAxNTUuNmMtMS4zLTMuNi0yLjctNy4zLTMuOS0xMC44IDI3IDMyLjYgNTQuMiA2NS40IDgwLjcgOTcuMiAxMy43IDE2LjUgMjcuNCAzMi45IDQxLjEgNDkuNSAyLjcgMy4xIDUuNCA2LjQgOCA5LjYgMy40IDQuMSA2LjggOC40IDEwLjUgMTIuNS0xLjIgMTAuMy0yLjIgMjAuNy0zLjMgMzAuNy0uNyA3LTEuNCAxNC0yLjIgMjEuMXYuMWMtLjMgNC41LS45IDktMS40IDEzLjQtLjcgNi4xLTIuMyAxOS45LTIuMyAxOS45bC0uMS43Yy0xLjggMjAuMi00IDQwLjYtNi4xIDYwLjQtLjkgOC4yLTEuNyAxNi42LTIuNiAyNS0uNS0xLjUtMS4xLTMtMS42LTQuNC0xLjUtNC0zLTguMi00LjQtMTIuM2wtMTAuNy0yOS43QzI0Mi45IDM0NC4yIDIwOSAyNTAgMTc1LjEgMTU1LjYiIGZpbGw9IiNmZmYiLz48L3N2Zz4=" /> TD5vrbiPgNCmM62FRxkgKmHpEHhWDVW6cb </a>

<a href="#polygon" alt="POLYGON"><img src="https://img.shields.io/badge/Donate-MATIC-7DFDFE?logo=data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBzdGFuZGFsb25lPSJubyI/Pgo8IURPQ1RZUEUgc3ZnIFBVQkxJQyAiLS8vVzNDLy9EVEQgU1ZHIDIwMDEwOTA0Ly9FTiIKICJodHRwOi8vd3d3LnczLm9yZy9UUi8yMDAxL1JFQy1TVkctMjAwMTA5MDQvRFREL3N2ZzEwLmR0ZCI+CjxzdmcgdmVyc2lvbj0iMS4wIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciCiB3aWR0aD0iNjAwLjAwMDAwMHB0IiBoZWlnaHQ9IjYwMC4wMDAwMDBwdCIgdmlld0JveD0iMCAwIDYwMC4wMDAwMDAgNjAwLjAwMDAwMCIKIHByZXNlcnZlQXNwZWN0UmF0aW89InhNaWRZTWlkIG1lZXQiPgoKPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsNjAwLjAwMDAwMCkgc2NhbGUoMC4xMDAwMDAsLTAuMTAwMDAwKSIKZmlsbD0iIzAwMDAwMCIgc3Ryb2tlPSJub25lIj4KPHBhdGggZD0iTTQyMDQgNTQ3OSBjLTIyIC00IC02NSAtMjIgLTk1IC00MCAtNzUgLTQ0IC0xNTYgLTkyIC0yNzQgLTE1OSAtMTE3Ci02NiAtMzYzIC0yMDggLTU5MCAtMzQwIC0yMjMgLTEyOSAtMzM5IC0xOTcgLTQxNCAtMjM5IC02OCAtMzkgLTExNyAtOTEgLTEzMgotMTQzIC01IC0yMCAtOSAtNTkwIC05IC0xNDI5IGwwIC0xMzk3IC0zNyAtMjIgYy04MCAtNDggLTIzMiAtMTM2IC0zMjggLTE5MAotNTUgLTMxIC0xMjcgLTczIC0xNjAgLTkyIC04NyAtNTEgLTI0OCAtMTQ0IC0zMzkgLTE5NiAtNzAgLTQwIC04MSAtNDMgLTEwMAotMzIgLTEyIDcgLTYxIDM3IC0xMTEgNjUgLTQ5IDI5IC0xNjYgOTYgLTI2MCAxNTAgLTkzIDU0IC0xOTUgMTEzIC0yMjUgMTMwCi0zMCAxNyAtMTEzIDY2IC0xODUgMTA3IGwtMTMwIDc1IC0zIDU0MCAtMiA1NDEgMjcgMTggYzE2IDEwIDY5IDQxIDExOCA2OSA1MAoyOSAxNjQgOTUgMjU1IDE0NyA5MSA1MiAyMzkgMTM4IDMzMCAxOTAgOTEgNTIgMTc0IDEwMCAxODYgMTA4IDIzIDE0IC0xIDI3CjM4MiAtMTk0IDExMiAtNjQgMjE1IC0xMjQgMjI4IC0xMzIgbDI0IC0xNiAwIDM2MyAtMSAzNjQgLTI2NiAxNTMgYy0zNTAgMjAyCi0zMzkgMjAxIC01NjMgNzEgLTY2IC0zNyAtMTk1IC0xMTIgLTI4NyAtMTY1IC05MiAtNTQgLTIwOCAtMTIxIC0yNTggLTE0OQotMTA2IC02MSAtNDc2IC0yNzQgLTU1NSAtMzIwIC0xOTEgLTExMSAtMjA0IC0xMjIgLTIyOCAtMTk0IC0xNyAtNTAgLTE3Ci0xNjQ4IDAgLTE3MDQgMTQgLTQ1IDYxIC05NSAxMjIgLTEzMCAyMyAtMTMgODQgLTQ5IDEzNiAtNzkgNTIgLTMwIDE3MCAtOTkKMjYzIC0xNTIgMjEwIC0xMjEgNDU2IC0yNjMgNjU3IC0zNzkgMjkyIC0xNzAgMjkwIC0xNjggMzQ1IC0xNzQgNzIgLTggOTkgNAoyOTggMTE5IDk0IDU1IDE5MCAxMTAgMjEyIDEyMyAyMiAxMiA2NyAzOCAxMDAgNTcgNzQgNDMgMTk5IDExNiA0MTUgMjQwIDkxCjUyIDIxMCAxMjEgMjY1IDE1MyA1NSAzMiAxMjAgNzAgMTQ1IDgzIDYyIDM1IDEwOSA4MCAxMzEgMTI3IDE4IDM4IDE5IDk4IDE5CjE0NDUgbDAgMTQwNSAxNzMgOTkgYzk0IDU0IDIwOCAxMjAgMjUyIDE0NiA0NCAyNiAxNjEgOTMgMjYwIDE1MCA5OSA1NyAxOTcKMTE0IDIxOCAxMjcgMzcgMjQgMzkgMjQgNjUgNyAxNSAtMTAgNzAgLTQyIDEyMiAtNzIgNTIgLTMwIDEyMCAtNjkgMTUwIC04NwozMCAtMTcgMTc3IC0xMDIgMzI1IC0xODcgMTQ5IC04NSAyODAgLTE2MSAyOTMgLTE2OSBsMjIgLTE0IDAgLTU0MCAwIC01NDEKLTEwMiAtNTkgYy01NyAtMzIgLTE0MSAtODEgLTE4OCAtMTA4IC00NyAtMjcgLTExNiAtNjggLTE1NSAtOTAgLTM4IC0yMiAtOTUKLTU1IC0xMjUgLTczIC0zMCAtMTcgLTk4IC01NiAtMTUwIC04NiAtNTIgLTMwIC0xMjMgLTcxIC0xNTYgLTkwIGwtNjIgLTM2Ci04MyA0NyBjLTQ2IDI2IC0xMjkgNzQgLTE4NCAxMDYgLTIwOCAxMjEgLTMzNyAxOTUgLTM0MSAxOTUgLTIgMCAtNCAtMTYyIC00Ci0zNTkgbDAgLTM1OSAxMTggLTY4IGMxNDggLTg1IDIwNSAtMTE4IDMxNyAtMTg1IDkwIC01NCA5MCAtNTQgMTc1IC01NCBsODUgMAoxMzUgNzkgYzc0IDQ0IDE3NiAxMDIgMjI1IDEzMSA1MCAyOCAxNjcgOTYgMjYwIDE1MCAzOTggMjMxIDQ4NCAyODAgNTUwIDMxOAoxNTcgODggMjEyIDEyMiAyNDIgMTUxIDY2IDYxIDY0IDIzIDYxIDk0OCBsLTMgODM2IC0zMCA0NCBjLTMzIDQ2IC02MiA2NgotMzEzIDIwOSAtMTE1IDY2IC0zMTYgMTgyIC01MTcgMjk5IC00NDMgMjU3IC02MTEgMzUxIC02MzIgMzU1IC01NCAxMSAtNzAgMTIKLTEwOSA0eiIvPgo8L2c+Cjwvc3ZnPgo=" /></a>
<a href="#bnb-bsc-20" alt="BNB"><img src="https://img.shields.io/badge/Donate-BEP--20-F3BA2F?logo=binance" /></a>
<a href="#eth" alt="ETH"><img src="https://img.shields.io/badge/Donate-Ethereum-51105E?logo=ethereum" /> 0xF18d0E57A9656EEd349dA54287bb33E1644d2f71 </a>

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
