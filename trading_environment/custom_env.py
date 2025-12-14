from collections import deque
import gymnasium as gym
import numpy as np
import pandas as pd

from trading_environment import utils
from trading_environment.rewards import RewardFunc



TIC_COL = 'tic'


class StockTradingEnv(gym.Env):
    """A custom implementation of a stock trading environment.
        - step() returns current data and past data from a window.

    Parameters
    ----------
    df : pd.DataFrame
        Contains stock prices and predictor data.
        Must be sorted by date (ascending) and stock ticker.
        Must have a 1:1 mapping between DataFrame Index values and date.
        Index values must be Integer type.
    predictors : list[str]
        List of predictors of stock prices.
    window_size : int
        Number of time steps (current and lagged) to track.
        Must be >= 1 (window_size = 1 means only track current state).
    integral_trades : bool
        Forces stock transactions to be integral.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        state_space: int,
        action_space: int,
        initial_amount: int,
        num_stock_shares: list[int],
        buy_cost_pct: list[float],
        sell_cost_pct: list[float],
        hmax: int,
        reward_scaling: float,
        reward_function: RewardFunc,
        price: str,
        stock_features: list[str],
        economy_features: list[str],
        window_size: int,
        integral_trades: bool=False,
        starting_step: int=None
    ):
        super().__init__()
        self.stock_dim = stock_dim
        self.state_space = state_space
        self.action_space = action_space
        self.initial_amount = initial_amount
        self.initial_shares = num_stock_shares
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.hmax = hmax
        self.reward_scaling = reward_scaling
        self.reward_function = reward_function
        self.price = price
        self.stock_features = stock_features
        self.economy_features = economy_features
        self.window_size = window_size
        self.integral_trades = integral_trades

        self.action_space = gym.spaces.Box(
            low=-1, 
            high=1, 
            shape=(self.action_space,), 
            dtype=np.float64
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.state_space),
            dtype=np.float64
        )
        self._terminal_step = None
        self._df = None
        self.n_tics = None
        self.tic_list = []
        self.nrow_per_tic = None
        self.state_history = None
        self.full_state_history = None
        self.starting_step = starting_step
        self.current_step = 0
        self.cash = self.initial_amount
        self.shares = np.array(
            self.initial_shares, 
            dtype=np.int64 if integral_trades else np.float64
        )

        self.df = df
        self._initialize_state()
        self.total_asset = self._compute_total_asset()

    @property
    def df(self):
        return self._df
    
    @df.setter
    def df(self, df: pd.DataFrame):
        # TO-DO: validate input df
        self._df = df
        self.tic_list = self._df[TIC_COL].unique().tolist()
        self.n_tics = len(self.tic_list)
        self.nrow_per_tic = int(self._df.shape[0] / self.n_tics)
        self._terminal_step = self.nrow_per_tic - 1

    @property
    def state(self):
        return self._get_current_state()
    
    def _get_current_prices(self) -> np.ndarray:
        """Returns a flat array of current prices."""
        prices = self.df.loc[self.current_step][[self.price]].values.reshape(-1)

        return prices
    
    def _get_current_stock_features(self) -> np.ndarray:
        """Returns a flat array of current stock features."""
        values = self.df.loc[self.current_step][self.stock_features].values.reshape(-1)

        return values
    
    def _get_current_economy_features(self) -> np.ndarray:
        """Returns a flat array of current economy features."""
        values = self.df.loc[self.current_step][self.economy_features].values
        # get first value (since economy features are the same for all stocks at time t)
        values = values[0]

        return values
    
    def _compute_total_asset(self) -> float:
        """Computes current total asset."""
        current_prices = self._get_current_prices()
        portfolio_worth = np.sum(self.shares * current_prices)
        total_asset = float(self.cash + portfolio_worth)

        return total_asset

    def _get_current_state(self) -> np.ndarray:
        """Returns a flat array of size self.state_space that contains data
        about the current state.
        """
        return np.concatenate(
            [
                np.array([self.cash]),
                self._get_current_prices(),
                self.shares,
                self._get_current_stock_features(),
                self._get_current_economy_features()
            ]
        )
    
    def _get_obs(self) -> np.ndarray:
        """Returns an array of size (self.window_size, self.state_space)
        that contains data about past and current states.
        """
        return np.array(self.state_history).astype(np.float64)
    
    def _initialize_state(self) -> None:
        """Adds self.window_size rows to state history and
        offsets self.current_step.
        """
        # sample starting step from deciles
        if self.starting_step is None:
            self.starting_step = utils.random_starting_step(
                total_steps=self.nrow_per_tic,
                interval=self.df.shape[0] // 10
            )

        self.state_history = deque(maxlen=self.window_size)
        self.full_state_history = np.empty((0, self.state_space))
        
        for _ in range(max(self.starting_step + 1, self.window_size)):
            state = self._get_current_state()
            self.state_history.append(state)
            self.full_state_history = np.concatenate(
                [self.full_state_history, state.reshape(1,-1)], axis=0)
            self.current_step += 1

        # required to log all initial states and not create a gap
        if self.current_step > 0:
            self.current_step -= 1

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_amount
        self.shares = np.array(
            self.initial_shares, 
            dtype=np.int64 if self.integral_trades else np.float64
        )
        self.total_asset = self._compute_total_asset()
        self._initialize_state()

        obs = self._get_obs()
        info = {}

        return obs, info
    
    def _buy(self, share_idx, action, price):
        cash_available = self.cash
        shares_to_buy = self.hmax * action
        if self.integral_trades:
            shares_to_buy = int(shares_to_buy)
        effective_price = price * (1 + self.buy_cost_pct[share_idx])

        if shares_to_buy * effective_price > cash_available:
            # clip to maximum shares affordable
            shares_bought = cash_available / effective_price
        else:
            shares_bought = shares_to_buy
        if self.integral_trades:
            shares_bought = int(shares_bought)

        # update states
        self.cash -= shares_bought * effective_price
        self.shares[share_idx] += shares_bought

    def _sell(self, share_idx, action, price):
        shares_available = self.shares[share_idx]
        shares_to_sell = self.hmax * (-action)
        if self.integral_trades:
            shares_to_sell = int(shares_to_sell)
        effective_price = price * (1 - self.sell_cost_pct[share_idx])

        if shares_to_sell > shares_available:
            # clip to maximum number of shares
            shares_sold = shares_available
        else:
            shares_sold = shares_to_sell

        # update states
        self.cash += shares_sold * effective_price
        self.shares[share_idx] -= shares_sold

    def step(self, actions):
        current_prices = self._get_current_prices()
        
        # execute transactions for each stock on current state s_t
        for i,tic in enumerate(self.tic_list):
            action = actions[i]
            price = current_prices[i]
            if action > 0:
                self._buy(i, action, price)
            elif action < 0:
                self._sell(i, action, price)

        # update internal state trackers after actions have been committed
        # i.e. get next state s_t+1
        self.current_step += 1
        self.total_asset = self._compute_total_asset()
        current_state = self._get_current_state()
        self.state_history.append(current_state)
        self.full_state_history = np.concatenate(
            [self.full_state_history, current_state.reshape(1,-1)], axis=0
        )
        
        # calculate reward as a function of {s_t+1, s_t, s_t-1, ...}
        reward = self.reward_function(self) * self.reward_scaling

        # return next state
        obs = self._get_obs()
        terminated = self.current_step >= self._terminal_step
        truncated = False
        info = {
            "cash": self.cash,
            "prices": current_prices,
            "shares": self.shares,
            "total_asset": self.total_asset
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        prices_str = " | ".join(f"{p:.2f}" for p in self._get_current_prices())
        print(
            f"Step: {self.current_step}, "
            f"Prices: {prices_str}, "
            f"Total Asset: {self.total_asset:.2f}"
        )
