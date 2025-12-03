import numpy as np


def simulate_series(
    T=500,                     # length of series
    c=0,                       # constant to add
    trend_slope=0.05,          # slope of linear trend
    noise_std=1.0,             # noise scale
    p=None,                    # Bernoulli probability
    freq=10,                   # frequency of indicator
    jump=5.0,                  # amount to add after delay
    duration=1,                # duration of the jump
    d1=2, d2=10,               # delay range: d ~ Uniform(a,b)
    seed=None
):
    if seed is not None:
        np.random.seed(seed)
    
    # --- 1. Linear upward trend ---
    trend = trend_slope * np.arange(T)
    
    # --- 2. Noise ---
    noise = np.random.normal(0, noise_std, T)
    
    # --- 3. Exogenous Bernoulli indicator ---
    if p is not None:
        indicator = np.random.binomial(1, p, T)
    else:
        indicator = np.zeros(T, dtype=int)
        indicator[::freq] = 1
    
    # delays for each activated indicator
    delays = np.random.randint(d1, d2+1, T)

    # array to accumulate delayed effects
    delayed_effect = np.zeros(T)

    for t in range(T):
        if indicator[t] == 1:
            d = delays[t]
            t_delayed = t + d
            if t_delayed + duration < T:
                for td in range(1,duration+1):
                    delayed_effect[t_delayed+td] += jump

    # --- Construct final series ---
    y = c + trend + noise + delayed_effect
    
    return y, indicator
