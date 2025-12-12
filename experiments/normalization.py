import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize


class MaskedVecNormalize(VecNormalize):
    def __init__(self, venv, normalize_mask, **kwargs):
        super().__init__(venv, **kwargs)
        normalize_mask = np.asarray(normalize_mask, dtype=bool)
        self.normalize_mask = normalize_mask

    def _normalize_obs(self, obs, info):
        if not self.norm_obs:
            return obs

        obs = obs.copy()
        mask = self.normalize_mask

        # mean are var have shapes (window_size, feature_dim)
        mean = self.obs_rms.mean
        var = self.obs_rms.var

        obs[..., mask] = (
            (obs[..., mask] - mean[:, mask]) /
            np.sqrt(var[:, mask] + self.epsilon)
        )
        return np.clip(obs, -self.clip_obs, self.clip_obs)