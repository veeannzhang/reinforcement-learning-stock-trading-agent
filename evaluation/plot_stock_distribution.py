plot_stock_distribution.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_stock_distribution(stock_idx, sorted_tickers, mean_base, std_base, mean_high, std_high):
    stock_name = sorted_tickers[stock_idx]

    # Parameters
    mu_base = mean_base[stock_idx]
    sigma_base = std_base[stock_idx]

    mu_high = mean_high[stock_idx]
    sigma_high = std_high[stock_idx]

    print(f"\n--- Stock: {stock_name} ---")
    print(f"Baseline: Mean={mu_base}, Std={sigma_base}")
    print(f"High VIX: Mean={mu_high}, Std={sigma_high}")
    shift = mu_high - mu_base
    print(f"Shift: {shift}")

    x_range = np.linspace(mu_base - 100, mu_base + 100, 2000)

    # Calculate PDFs
    pdf_base = norm.pdf(x_range, mu_base, sigma_base)
    pdf_high = norm.pdf(x_range, mu_high, sigma_high)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Standard View (Wide)
    ax1.plot(x_range, pdf_base, label='Baseline', color='blue', linestyle='-')
    ax1.plot(x_range, pdf_high, label='High VIX', color='red', linestyle='--', alpha=0.7)
    ax1.set_title(f'Overlay of Policy Distributions for {stock_name}')
    ax1.set_xlabel('Action Logit Value')
    ax1.set_ylabel('Probability Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Extreme Zoom on the Means
    # Dynamic Zoom Window
    # Make the window large enough to hold the shift, plus some padding
    dynamic_window = max(0.1, abs(shift) * 2.0)

    # Center the plot between the two means for a better view
    midpoint = (mu_base + mu_high) / 2
    x_zoom = np.linspace(midpoint - dynamic_window, midpoint + dynamic_window, 10000)

    pdf_base_zoom = norm.pdf(x_zoom, mu_base, sigma_base)
    pdf_high_zoom = norm.pdf(x_zoom, mu_high, sigma_high)

    ax2.plot(x_zoom, pdf_base_zoom, label='Baseline Mean', color='blue')
    ax2.plot(x_zoom, pdf_high_zoom, label='High VIX Mean', color='red', linestyle='--')

    # Add vertical lines to mark the exact centers
    ax2.axvline(mu_base, color='blue', linestyle=':', alpha=0.5, label=f'Base Mean: {mu_base:.3f}')
    ax2.axvline(mu_high, color='red', linestyle=':', alpha=0.5, label=f'High Mean: {mu_high:.3f}')

    ax2.set_title(f'Extreme Zoom: Visualizing the {shift:.4f} Shift')
    ax2.set_xlabel('Action Logit Value')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()