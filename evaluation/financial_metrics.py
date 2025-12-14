import numpy as np
import pandas as pd


def net_asset_change(df: pd.DataFrame):
    """Calculate change in net assets as the change in 
    total cash and portfolio worth.

    Arguments:
    ----------
        df : pd.DataFrame
            transactions DataFrame containing cash, prices, and shares
            at each time step. Output of run_backtest() function.
    """
    # differences from first to last state
    diff = df.iloc[-1] - df.iloc[0]

    # calculate changes in different assets
    prices_diff = diff[[col for col in df.columns if 'price' in col]].values
    shares_diff = diff[[col for col in df.columns if 'shares' in col]].values
    cash_diff = diff['cash']

    asset_change = sum(prices_diff * shares_diff) + cash_diff

    return asset_change


def get_total_return(asset_per_step: np.ndarray) -> float:
    """Calculate total return"""
    start_assets = asset_per_step[0]
    end_assets = asset_per_step[-1]
    
    total_return = (end_assets - start_assets) / start_assets

    return total_return


def get_cagr(asset_per_step: np.ndarray, days: int) -> float:
    """Calculate CAGR"""
    start_assets = asset_per_step[0]
    end_assets = asset_per_step[-1]
    
    cagr = (end_assets / start_assets) ** (365 / days) - 1

    return cagr


def _sharpe_ratio(returns: np.ndarray) -> float:
    """Calculates Sharpe ratio on an array of returns"""
    mean_r = np.mean(returns)
    std_r = np.std(returns, ddof=1)

    return np.sqrt(365) * mean_r / std_r


def get_sharpe_ratio(asset_per_step: np.ndarray) -> float:
    """Calculate Sharpe Ratio"""
    returns_per_step = asset_per_step[1:] / asset_per_step[:-1] - 1

    return _sharpe_ratio(returns_per_step)


def get_volatility(asset_per_step: np.ndarray) -> float:
    """Calculate volatility in returns"""
    returns_per_step = asset_per_step[1:] / asset_per_step[:-1] - 1

    return np.sqrt(365) * np.std(returns_per_step, ddof=1)

    
def get_var(asset_per_step: np.ndarray, alpha=0.05) -> float:
    """Calculate Value-at-Risk (VaR)"""
    r = asset_per_step[1:] / asset_per_step[:-1] - 1

    return -np.quantile(r, alpha)


def get_turnover(
        cash: np.ndarray,
        prices: np.ndarray,
        shares: np.ndarray
    ) -> float:
    """Calculate turnover"""
    asset_per_step = cash + (prices * shares).sum(axis=1)
    w = (prices * shares) / asset_per_step[:, None]

    return np.mean(np.sum(np.abs(np.diff(w, axis=0)), axis=1)) / 2


def financial_performance(transactions_df: pd.DataFrame) -> dict:
    """Returns a set of financial performance metrics
    Arguments:
    ----------
        transactions_df : pd.DataFrame
            output of run_backtest() function
    """
    price_cols = [col for col in transactions_df.columns if 'price' in col]
    share_cols = [col for col in transactions_df.columns if 'share' in col]

    cash = transactions_df['cash'].values
    prices = transactions_df[price_cols].values
    shares = transactions_df[share_cols].values

    asset_per_step = cash + (prices * shares).sum(axis=1)

    metrics = {
        'total_return': get_total_return(asset_per_step),
        'CAGR': get_cagr(asset_per_step, days=len(prices)),
        'volatility': get_volatility(asset_per_step),
        'Sharpe': get_sharpe_ratio(asset_per_step),
        'VaR': get_var(asset_per_step, alpha=0.05),
        'turnover': get_turnover(cash, prices, shares)
    }

    return metrics
