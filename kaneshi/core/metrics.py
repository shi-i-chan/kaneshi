import numpy as np

from kaneshi.config import DEF_COMMISSION

from numpy.typing import NDArray


# TODO add Sharp ratio
# TODO add Sortino ratio
# TODO add Calmar ratio
# TODO add more metrics


def n_trades(strategy) -> int:
    """ Get number of trades """
    return len(strategy.trades)


def max_drawdown(strategy) -> float:
    """ Get maximum single drawdown """
    return np.amin(strategy.trades)


def max_return(strategy) -> float:
    """ Get maximum trade profit"""
    return np.amax(strategy.trades)


def win_rate(strategy) -> float:
    """ Get good/bad trades ratio """
    return (strategy.trades > 0).sum() / strategy.trades.shape[0]


def mean_trade_time(strategy) -> float:
    """ Get mean trade time in minutes """
    diff = np.array([(sell - buy).astype('timedelta64[m]')
                     for buy, sell in zip(strategy.buy_indices, strategy.sell_indices)])
    return diff.astype('int').mean()


def equal_return(strategy=None, trades: NDArray = None, start_dep: int = 100) -> float:
    """ Get total return with equal deposit"""
    trades = strategy.trades if trades is None else trades
    dep = np.full_like(trades, start_dep)
    dep = dep * (1 - DEF_COMMISSION)
    dep = dep + dep * trades
    dep = dep * (1 - DEF_COMMISSION)
    dep = dep - start_dep
    return sum(dep)


def cumulative_return(strategy=None, trades: NDArray = None, start_dep: int = 100) -> float:
    """ Get total return with cumulative deposit """
    dep = start_dep
    trades = strategy.trades if trades is None else trades
    for trade in trades:
        dep = dep * (1 - DEF_COMMISSION)
        dep = dep + dep * float(trade)
        dep = dep * (1 - DEF_COMMISSION)
    return dep - start_dep
