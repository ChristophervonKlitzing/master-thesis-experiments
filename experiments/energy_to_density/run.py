from typing import List
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")  # Clean look for presentations

def u(x: np.ndarray) -> np.ndarray:
    """Quadratic energy with three local minima."""
    base = 0.04 * x**2
    well1 = 3.0 * np.exp(-((x + 4.0) / 0.4) ** 2)
    well2 = 1.2 * np.exp(-((x + 2.5) / 0.9) ** 2)
    well3 = 2.0 * np.exp(-((x - 2.0) / 2.0) ** 2)
    return base - (well1 + well2 + well3)

def p(x: np.ndarray) -> np.ndarray:
    """Unnormalized density: p(x) = exp(-u(x))."""
    return np.exp(-u(x))

def plot_energy_and_density():
    """Generate and display u(x) and p(x) on shared x-axis subplots."""
    x = np.linspace(-8, 6, 301)
    ux = u(x)
    px = p(x)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax: List[Axes]
    fontsize = 20

    # Plot u(x)
    # ax[0].plot(x, np.zeros_like(x), linestyle="--", color="lightgray")
    ax[0].plot(x, ux, color='navy', linewidth=2)
    # ax[0].set_title("Energy Function $u(x)$", fontsize=14)
    ax[0].set_ylabel("$u(x)$", fontsize=fontsize, rotation=0)
    ax[0].yaxis.set_label_coords(-0.05, 0.7)
    ax[0].set_yticks([])
    ax[0].set_xlabel("$x$", fontsize=fontsize)
    ax[0].xaxis.set_label_coords(1.0, 0.4)
    # ax[0].tick_params(axis='both', which='major', labelsize=fontsize)
    
    
    ux_margin = (ux.max() - ux.min()) * 0.1
    ax[0].set_ylim(ux.min() - ux_margin, ux.max() + ux_margin)
    ax[0].grid(False)
    

    # Plot p(x)
    ax[1].plot(x, px, color='green')
    ax[1].fill_between(x, px, color="green", alpha=0.2)
    # ax[1].set_title("Unnormalized Density $\\tilde{p}(x) = e^{-u(x)}$", fontsize=14)
    ax[1].set_ylabel("$\\tilde{p}(x)$", fontsize=fontsize, rotation=0)
    ax[1].yaxis.set_label_coords(-0.05, 0.7)
    ax[1].set_yticks([])
    ax[1].set_xlabel("$x$", fontsize=fontsize)
    ax[1].xaxis.set_label_coords(1.0, -0.1)
    px_margin = (px.max() - px.min()) * 0.1
    ax[1].set_ylim(px.min(), px.max() + px_margin)
    ax[1].grid(False)
    
    spine_color = "black"
    for i in [0, 1]:
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines["left"].set_color(spine_color)
        ax[i].spines["bottom"].set_color(spine_color)

        ax[i].set_xticks([])

        x_offset = 0.0
        y_offset = x_offset / ax[i].get_window_extent().height * ax[i].get_window_extent().width

        ax[i].spines.top.set_visible(False)
        ax[i].spines.right.set_visible(False)
        ax[i].spines.left.set_position(('axes', -x_offset))
        if i == 0:
            ax[i].spines.bottom.set_position(('axes', 0.5))
            ax[i].plot(1, 0.5, ">k", transform=ax[i].transAxes, clip_on=False, color=spine_color)
        else:
            ax[i].spines.bottom.set_position(('axes', -y_offset))
            ax[i].plot(1, -y_offset, ">k", transform=ax[i].transAxes, clip_on=False, color=spine_color)
        ax[i].plot(-x_offset, 1, "^k", transform=ax[i].transAxes, clip_on=False, color=spine_color)
    
    # ax[1].set_visible(False)
    plt.show()



def run(args):
    # Run the visualization
    plot_energy_and_density()
