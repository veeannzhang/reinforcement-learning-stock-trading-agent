import pandas as pd


def run_backtest(model, env, env_config, deterministic=True):
    """
    model : SB3 model
    env   : VecNormalize(DummyVecEnv([make_env]))  (what you're already using)
    env_config : dict with at least {"stock_dim": int}

    Assumes:
      - Gymnasium-style underlying env, recording `full_state_history`
      - Underlying env has `tic_list` and `full_state_history` attributes
      - VecEnv interface:
          obs = env.reset()
          obs, reward, terminated, truncated, info = env.step(action)
    """
    # ----- roll out one full episode on the vectorized env
    # get underlying env by removing all extra wrappers (Monitor, TimeLimit, etc.)
    base_env = env.venv.envs[0]
    while hasattr(base_env, "env"):
        base_env = base_env.env
    
    # initialize
    obs = env.reset()
    full_state_history = base_env.full_state_history

    terminated = False
    while not terminated:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated_arr, info = env.step(action)
        terminated = bool(terminated_arr[0])

        # copy latest state history
        # avoid overwriting with reset version (at final while loop)
        if base_env.full_state_history.shape[0] > full_state_history.shape[0]:
            full_state_history = base_env.full_state_history

    # ----- build dataframe from rstored state history
    stock_dim = env_config.get("stock_dim")
    lim = 1 + 2 * stock_dim  # cash + prices + shares

    full_state_df = pd.DataFrame(full_state_history[:, :lim])
    full_state_df.columns = (
        ["cash"]
        + [f"price_{tic}" for tic in base_env.tic_list]
        + [f"shares_{tic}" for tic in base_env.tic_list]
    )

    return full_state_df