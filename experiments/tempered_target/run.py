from typing import List
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def run(args=None):
    # Define x-axis
    x = np.linspace(-6, 6, 200)

    # Define two-mode target distribution p(x)
    p = (0.6 * norm.pdf(x, loc=-2, scale=0.3) + 0.4 * norm.pdf(x, loc=2, scale=0.3))

    temperatures = [1.0, 4.0, 16.0]

    # Set up horizontal subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    axes: List[Axes]

    fontsize = 25
    file_type = "pgf"

    for ax, T in zip(axes, temperatures):
        # Temper the distribution
        p_T = p ** (1/T)
        p_T /= np.trapz(p_T, x)  # Normalize

        # Plot
        ax.plot(x, p_T, label=f'T = {T}', color='blue')

        if T > 1:
            ax.plot(x, p, '--', color='gray', label='Original p(x)')
        ax.fill_between(x, p_T, color='cornflowerblue', alpha=0.4)
        # ax.set_title(f'T = {T}')
        ax.text(-0.9, 0.63, f"T = {T:.0f}", color="blue", fontsize=fontsize)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['bottom'].set_position(('outward', 10))

        ax.get_yaxis().set_visible(False)
        ax.tick_params(left=False, bottom=False, labelbottom=False)
        ax.axis('off')  # Optional: keep only if you don't want any axes

    # Layout and display
    fig.tight_layout()
    fig.savefig(f"outputs/experiments/tempered_target.{file_type}", format=file_type, bbox_inches='tight')
    plt.close(fig)