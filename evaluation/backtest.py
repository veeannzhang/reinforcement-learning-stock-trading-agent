import pandas as pd


def run_backtest(
        model, 
        env, 
        env_config, 
        deterministic=True
    ) -> pd.DataFrame:
    """
    Arguments:
    ----------
        model : SB3 model
        env   : VecNormalize(DummyVecEnv([make_env]))
        env_config : dict with at least {"stock_dim": int}

    Assumptions:
      - Gymnasium environment, with `full_state_history` attribute
      - Underlying env has `tic_list` and `full_state_history` attributes
      - VecEnv interface:
          obs = env.reset()
          obs, reward, terminated, truncated, info = env.step(action)
    """
    # ----- rollout one full episode on the vectorized env
    # get underlying env by removing all extra wrappers (Monitor, TimeLimit, etc.)
    try:
        # work-around to accept DummyVecEnv without VecNormalize
        base_env = env.venv.envs[0]
        while hasattr(base_env, "env"):
            base_env = base_env.env
    except AttributeError:
        base_env = env.envs[0]
    
    # initialize
    obs = env.reset()
    full_state_history = base_env.full_state_history
    # store rewards
    rewards = []

    terminated = False
    while not terminated:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated_arr, info = env.step(action)
        rewards.append(float(reward[0]))
        terminated = bool(terminated_arr[0])

        # copy latest state history
        # avoid overwriting with reset version (at final while loop)
        if base_env.full_state_history.shape[0] > full_state_history.shape[0]:
            full_state_history = base_env.full_state_history

    # ----- build dataframe from stored state history
    stock_dim = env_config.get("stock_dim")
    lim = 1 + 2 * stock_dim  # cash + prices x stock_dim + shares x stock_dim
    full_state_df = pd.DataFrame(full_state_history[:, :lim])
    full_state_df.columns = (
        ["cash"]
        + [f"price_{tic}" for tic in base_env.tic_list]
        + [f"shares_{tic}" for tic in base_env.tic_list]
    )
    # add rewards column
    full_state_df['reward'] = 0 # initialize column
    full_state_df.iloc[-len(rewards):, -1] = rewards

    return full_state_df