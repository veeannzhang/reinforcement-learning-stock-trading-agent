import numpy as np


def simulate_series(
    T=500,
    trend_constant=100,
    trend_slope=0.01,
    amp=3,
    period=100,
    noise_std=1,
):
    time_idx = np.arange(T)

    # simulate price components
    seasonality = amp * np.cos(2 * np.pi * time_idx / period)
    trend = trend_constant + trend_slope * time_idx
    noise = np.random.normal(0, noise_std, size=T)

    y = seasonality + trend + noise

    return y