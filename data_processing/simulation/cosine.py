import numpy as np


def simulate_series(
    T: int=500,
    trend_constant: float=100.0,
    trend_slope: float=0.01,
    amp: float=3.0,
    period: int=100,
    noise_std: float=1.0,
):
    """Simulates a linear trend with cosine seasonality.

    Arguments:
    ----------
        T : int
            Number of time steps.
        trend_constant : int
            Constant component of linear trend.
        trend_slope : int
            Slope of linear trend.
        amp : float
            Amplitude of cosine.
        period : int
            Period of cosine.
        noise_std : float
            Standard deviation of random Gaussian noise.
    """
    time_idx = np.arange(T)

    # simulate price components
    seasonality = amp * np.cos(2 * np.pi * time_idx / period)
    trend = trend_constant + trend_slope * time_idx
    noise = np.random.normal(0, noise_std, size=T)

    y = seasonality + trend + noise

    return y