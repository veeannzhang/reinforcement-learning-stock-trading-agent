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