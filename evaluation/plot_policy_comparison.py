import numpy as np
import matplotlib.pyplot as plt

def plot_policy_comparison(tickers, mean_base, mean_high, std_base, std_high):
    """
    Plots the comparison of mean actions and standard deviations for baseline vs high VIX scenarios.
    """
    num_stocks = len(tickers)
    indices = np.arange(num_stocks)
    width = 0.35

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 12))

    # Plot Means
    ax = axes[0]
    rects1 = ax.bar(indices - width/2, mean_base, width, label='Baseline (Avg VIX)', color='skyblue')
    rects2 = ax.bar(indices + width/2, mean_high, width, label='High VIX (95th %ile)', color='salmon')

    ax.set_ylabel('Mean Action Value (Logits)')
    ax.set_title('Comparison of Mean Actions: Baseline vs High VIX')
    ax.set_xticks(indices)
    ax.set_xticklabels(tickers)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Plot Standard Deviations
    ax = axes[1]
    rects3 = ax.bar(indices - width/2, std_base, width, label='Baseline (Avg VIX)', color='skyblue')
    rects4 = ax.bar(indices + width/2, std_high, width, label='High VIX (95th %ile)', color='salmon')

    ax.set_ylabel('Standard Deviation')
    ax.set_title('Comparison of Action Uncertainty (Std): Baseline vs High VIX')
    ax.set_xticks(indices)
    ax.set_xticklabels(tickers)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()
