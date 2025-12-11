import numpy as np
import pandas as pd


def _unwrap_env(env):
    """
    Unwrap Monitor / TimeLimit / other wrappers until we reach the
    underlying StockTradingEnv that actually has full_state_history, etc.
    """
    # SB3 Monitor and many gym wrappers expose the wrapped env as .env
    while hasattr(env, "env"):
        env = env.env
    return env


def run_backtest(model, env, env_config):
    """
    Apply the trained model to a trading environment (plain or VecEnv)
    and log actions.

    Parameters
    ----------
    model : SB3 model
        Trained RL model with .predict().
    env :
        Trading environment. Can be:
        - StockTradingEnv (gym.Env-like), or
        - DummyVecEnv wrapping StockTradingEnv (optionally via Monitor).
    env_config : dict
        Configuration containing at least 'stock_dim'.
    """

    # Detect if this is a VecEnv (e.g. DummyVecEnv)
    is_vec_env = hasattr(env, "envs")

    if is_vec_env:
        # VecEnv: observations and dones are batched
        obs = env.reset()              # shape: (n_envs, obs_dim), here n_envs=1
        done = np.array([False])

        # We only have one env, so it's env.envs[0]
        base_env = _unwrap_env(env.envs[0])

        while not done[0]:
            # obs already has batch dimension; model.predict is fine with that
            actions, _ = model.predict(obs, deterministic=True)
            obs, rewards, done, infos = env.step(actions)

    else:
        # Non-vectorized, Gymnasium-style env
        obs, info = env.reset()
        terminated = False
        truncated = False

        base_env = _unwrap_env(env)

        while not (terminated or truncated):
            actions, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(actions)

    # Build result dataframe from the underlying StockTradingEnv
    stock_dim = env_config.get('stock_dim')
    lim = 1 + 2 * stock_dim

    res = pd.DataFrame(base_env.full_state_history[:, :lim])
    res.columns = [
        'cash',
        *[f'price_{tic}' for tic in base_env.tic_list],
        *[f'shares_{tic}' for tic in base_env.tic_list],
    ]

    return res
