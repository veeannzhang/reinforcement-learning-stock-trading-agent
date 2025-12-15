import numpy as np
import torch

def get_policy_distributions(
    model,
    valid_norm_env,
    train_df,
    vix_col="VIXCLS",
    vix_idx=66,
    high_sigma=1.5,
):
    device = model.device
    n_envs = valid_norm_env.num_envs

    # VIX normalization stats
    vix_data = train_df[vix_col].to_numpy().reshape(-1, 1)
    vix_mean = float(np.mean(vix_data))
    vix_std = float(np.std(vix_data)) + 1e-8
    high_vix_raw = vix_mean + high_sigma * (vix_std - 1e-8)

    # Reset ONCE so only VIX differs
    obs0 = valid_norm_env.reset()

    obs_base = obs0.copy()
    obs_base[:, -1, vix_idx] = (vix_mean - vix_mean) / vix_std  # 0

    obs_high = obs0.copy()
    obs_high[:, -1, vix_idx] = (high_vix_raw - vix_mean) / vix_std

    obs_base_tensor = torch.as_tensor(obs_base, device=device, dtype=torch.float32)
    obs_high_tensor = torch.as_tensor(obs_high, device=device, dtype=torch.float32)

    is_recurrent = hasattr(model.policy, "lstm_actor")

    with torch.no_grad():
        if is_recurrent:
            # RecurrentPPO path
            n_lstm_layers = model.policy_kwargs.get("n_lstm_layers", 1)
            lstm_hidden_size = model.policy.lstm_actor.hidden_size
            initial_lstm_states = (
                torch.zeros(n_lstm_layers, n_envs, lstm_hidden_size, device=device),
                torch.zeros(n_lstm_layers, n_envs, lstm_hidden_size, device=device),
            )
            episode_starts = torch.ones(n_envs, device=device, dtype=torch.float32)

            dist_base, _ = model.policy.get_distribution(
                obs_base_tensor, initial_lstm_states, episode_starts
            )
            dist_high, _ = model.policy.get_distribution(
                obs_high_tensor, initial_lstm_states, episode_starts
            )
        else:
            # Standard PPO path (your MambaFeatureExtractor PPO)
            dist_base = model.policy.get_distribution(obs_base_tensor)
            dist_high = model.policy.get_distribution(obs_high_tensor)

    # Continuous actions (DiagGaussianDistribution)
    mean_base = dist_base.distribution.mean.detach().cpu().numpy().reshape(-1)
    std_base  = dist_base.distribution.stddev.detach().cpu().numpy().reshape(-1)
    mean_high = dist_high.distribution.mean.detach().cpu().numpy().reshape(-1)
    std_high  = dist_high.distribution.stddev.detach().cpu().numpy().reshape(-1)

    return mean_base, std_base, mean_high, std_high, vix_mean, high_vix_raw