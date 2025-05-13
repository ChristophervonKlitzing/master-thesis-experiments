from typing import List
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def run(args=None):
    # Define x-axis
    x = np.linspace(-6, 6, 100)

    # Define two-mode target distribution p(x)
    p = 0.5 * norm.pdf(x, loc=-2, scale=0.7) + 0.5 * norm.pdf(x, loc=2, scale=0.7)

    # I-projection: covers one mode
    q_I = norm.pdf(x, loc=-2, scale=0.7)

    # M-projection: covers both modes
    q_M = norm.pdf(x, loc=0, scale=2.2)

    fontsize = 25

    file_type = "pgf"

    # Projections to plot separately
    projections = [
        (q_I, f"i_projection.{file_type}", lambda ax: ax.text(-3.2, 0.45, "$q$", fontsize=fontsize, color="blue")),
        (q_M, f"m_projection.{file_type}", lambda ax: ax.text(0, 0.22, "$q$", fontsize=fontsize, color="blue")),
    ]

    for q, filename, annotate_q in projections:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax: Axes
        ax.set_ylim(0, 0.6)

        ax.plot(x, p, color='green', linewidth=1)
        ax.plot(x, q, color='blue', linewidth=1)
        ax.fill_between(x, p, color='green', alpha=0.2)
        ax.fill_between(x, q, color='cornflowerblue', alpha=0.4)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['bottom'].set_position(('outward', 10))

        ax.get_yaxis().set_visible(False)
        ax.tick_params(left=False, bottom=False, labelbottom=False)

        ax.axis('off')  # Optional: keep only if you don't want any axes

        ax.text(2.5, 0.3, "$p^*$", fontsize=fontsize, color="green")
        annotate_q(ax)

        fig.tight_layout()
        fig.savefig(f"outputs/experiments/i_and_m_projection/{filename}", format=file_type, bbox_inches='tight')
        plt.close(fig)
