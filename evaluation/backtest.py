import pandas as pd
import numpy as np

def run_backtest(model, env, env_config):
    """Apply the trained model to a trading environment and log actions.

    Parameters
    ----------
    model : SB3 model
        Trained RL model with .predict().
    env:
        Trading environment.
    env_config:

    """
    # get starting state
    obs, info = env.reset()

    t = env.current_step
    terminated = False
    truncated = False

    while not (terminated or truncated):
        state = env.state
        actions, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(actions)

    # exclude predictors data
    stock_dim = env_config.get('stock_dim')
    lim = 1 + 2 * stock_dim
    res = pd.DataFrame(np.array(env.state_history)[:,:lim])    
    res.columns = [
        'cash',
        *['price_' + tic for tic in env.tic_list],
        *['shares_' + tic for tic in env.tic_list]
    ]

    return res