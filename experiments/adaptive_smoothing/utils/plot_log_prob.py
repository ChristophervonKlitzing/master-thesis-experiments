from typing import Callable
from matplotlib.axes import Axes
import torch


def plot_probability(
        ax: Axes, 
        log_prob: Callable[[torch.Tensor], torch.Tensor], 
        title: str, 
        min_x: float = -4., 
        min_y: float = -4., 
        max_x: float = 4., 
        max_y: float = 4.
    ) -> None:

    resolution = 100

    with torch.no_grad():
        xline = torch.linspace(min_x, max_x, resolution)
        yline = torch.linspace(min_y, max_y, resolution)
        xgrid, ygrid = torch.meshgrid(xline, yline)
        xyinput = torch.hstack([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)])
        
        zgrid = log_prob(xyinput)
        zgrid = torch.reshape(zgrid, (resolution, resolution))
        zgrid = torch.exp(zgrid)
        
        ax.contourf(xgrid, ygrid, zgrid, levels=100)
        ax.set_title(title)