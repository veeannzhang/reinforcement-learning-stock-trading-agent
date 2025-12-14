import torch
import numpy as np

def get_policy_distributions_for_vix_scenarios(
    model,
    valid_norm_env,
    train_df,
    vix_col="VIXCLS",
    vix_idx=66,
):
    device = model.device
    n_envs = valid_norm_env.num_envs

    # LSTM sizes
    n_lstm_layers = model.policy_kwargs.get("n_lstm_layers", 1)
    lstm_hidden_size = model.policy.lstm_actor.hidden_size

    # Initial LSTM states
    initial_lstm_states = (
        torch.zeros(n_lstm_layers, n_envs, lstm_hidden_size, device=device),
        torch.zeros(n_lstm_layers, n_envs, lstm_hidden_size, device=device),
    )

    # IMPORTANT: float32, not bool (PyTorch disallows 1.0 - bool)
    episode_starts = torch.ones(n_envs, device=device, dtype=torch.float32)

    # VIX normalization stats (from training df)
    vix_data = train_df[vix_col].to_numpy().reshape(-1, 1)
    vix_mean = float(np.mean(vix_data))
    vix_std = float(np.std(vix_data)) + 1e-8

    # Baseline obs: set VIX at mean (normalized = 0)
    obs_base = valid_norm_env.reset()
    obs_base[:, -1, vix_idx] = (vix_mean - vix_mean) / vix_std
    obs_base_tensor = torch.as_tensor(obs_base, device=device, dtype=torch.float32)

    # High VIX obs: +1.5 std
    high_vix_raw = vix_mean + 1.5 * (vix_std - 1e-8)  # undo the +1e-8 for the raw value
    obs_high = valid_norm_env.reset()
    obs_high[:, -1, vix_idx] = (high_vix_raw - vix_mean) / vix_std
    obs_high_tensor = torch.as_tensor(obs_high, device=device, dtype=torch.float32)

    with torch.no_grad():
        dist_base, _ = model.policy.get_distribution(
            obs_base_tensor, initial_lstm_states, episode_starts
        )
        dist_high, _ = model.policy.get_distribution(
            obs_high_tensor, initial_lstm_states, episode_starts
        )

    mean_base = dist_base.distribution.mean.detach().cpu().numpy().reshape(-1)
    std_base = dist_base.distribution.stddev.detach().cpu().numpy().reshape(-1)
    mean_high = dist_high.distribution.mean.detach().cpu().numpy().reshape(-1)
    std_high = dist_high.distribution.stddev.detach().cpu().numpy().reshape(-1)

    return mean_base, std_base, mean_high, std_high, vix_mean, high_vix_raw