import numpy as np
from dataclasses import dataclass
from typing import Protocol


def _get_values_from_history(
        hist: np.ndarray, 
        t: int,
        stock_dim: int
    ):
    """Helper function that returns cash, prices, and shares at timestep t
    from the state history of the trading environment.

    Arguments:
    ----------
        hist : np.ndarray
            The state history array.
        t : int
            Timestep.
        stock_dim : int
            Number of stocks in the environment.
    """
    # cash is tracked at idx = 0
    cash = hist[t, 0]

    # prices
    prices = hist[t, 1:stock_dim + 1]

    # shares
    shares = hist[t, stock_dim+1:stock_dim*2+1]

    return cash, prices, shares


class RewardFunc(Protocol):
    def __call__(self, env) -> float:
        pass


class PnLReward:
    """Calculates change in net asset (cash and portfolio value)"""

    def __call__(self, env) -> float:
        hist = env.full_state_history
        n_tics = env.n_tics

        cash_prev, prices_prev, shares_prev = _get_values_from_history(
            hist, t=-2, stock_dim=n_tics
        )
        cash_now, prices_now, shares_now = _get_values_from_history(
            hist, t=-1, stock_dim=n_tics
        )

        cash_change = cash_now - cash_prev
        portfolio_change =\
            (prices_now * shares_now).sum() - (prices_prev * shares_prev).sum()
        net_asset_change = cash_change + portfolio_change

        return net_asset_change
    

class PenalizedTurnover:
    """Penalizes large asset changes, rewards asset growth.
    Assumption: NO SHORTING
    """
    def __init__(self, penalty: float):
        self.penalty = penalty

    def __call__(self, env) -> float:
        hist = env.full_state_history
        n_tics = env.n_tics

        cash_prev, prices_prev, shares_prev = _get_values_from_history(
            hist, t=-2, stock_dim=n_tics
        )
        cash_now, prices_now, shares_now = _get_values_from_history(
            hist, t=-1, stock_dim=n_tics
        )
        value_per_share_prev = prices_prev * shares_prev
        value_per_share_now = prices_now * shares_now

        # log asset growth
        v_prev = value_per_share_prev.sum() + cash_prev
        v_now = value_per_share_now.sum() + cash_now
        if v_prev <= 0 or v_now <= 0:
            log_growth = 0.0
        else:
            log_growth = float(np.log(v_now / v_prev))

        # L2-norm of portfolio & cash change
        w_prev = np.append(value_per_share_prev, cash_prev) / v_prev
        w_now = np.append(value_per_share_now, cash_now) / v_now
        turnover = float(np.linalg.norm(w_now - w_prev, ord=2))

        # reward asset growth, penalize portfolio change
        reward = log_growth - self.penalty * turnover

        return reward