import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize


class MaskedVecNormalize(VecNormalize):
    """Stable-baselines3 VecNormalize adapted to normalize
    only selected features.

    Parameters:
    -----------
        normalize_mask : np.ndarray
            Flat boolean array of dim = features_dim.
            True at the indices of features that needs to be normalized,
            False otherwise.
    """
    def __init__(self, venv, normalize_mask, **kwargs):
        super().__init__(venv, **kwargs)
        normalize_mask = np.asarray(normalize_mask, dtype=bool)
        self.normalize_mask = normalize_mask

    def _normalize_obs(self, obs, info):
        if not self.norm_obs:
            return obs

        obs = obs.copy()
        mask = self.normalize_mask

        # mean and var have shapes (window_size, feature_dim)
        mean = self.obs_rms.mean
        var = self.obs_rms.var

        # selective normalization
        obs[..., mask] = (
            (obs[..., mask] - mean[:, mask]) /
            np.sqrt(var[:, mask] + self.epsilon)
        )
        # selective clipping
        obs[..., mask] = np.clip(obs[..., mask], -self.clip_obs, self.clip_obs)

        return obs