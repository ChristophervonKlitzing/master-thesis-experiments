import copy
from functools import partial
from typing import Callable, List, Tuple
from matplotlib.axes import Axes
import zuko
import torch 
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt 
import tempfile
import os 
from PIL import Image

from .targets.energy_potentials import U1
from .utils.plot_log_prob import plot_probability


def log_Z(eta: float, reward: torch.Tensor) -> torch.Tensor:
    log_num_samples = torch.tensor(reward.shape[0]).log()
    log_Z: torch.Tensor = torch.logsumexp(reward / (1 + eta), dim=0) - log_num_samples
    return log_Z

def dual_impl(eta: float, epsilon: float, reward: torch.Tensor):
    dual = -(1 + eta) * log_Z(eta, reward) - eta * epsilon
    return dual.item()

def find_best_eta(dual: Callable[[float], float]) -> float:
    # dual is a convex 1D maximization problem -> turn into minization problem
    f = lambda eta: -dual(eta)
    res = minimize_scalar(f, bounds=(0, 1e6), method="bounded")
    best_eta: float = res.x
    return best_eta

def adaptive_smoothing_training(
        target_log_prob_fn: Callable[[torch.Tensor], torch.Tensor], 
        num_iterations: int, 
        eval_interval: int, 
        flow: zuko.flows.Flow,
        epsilon: float,
        num_q_bar_samples: int = 1000,
        out_gif_path: str = "outputs/adaptive_smoothing.gif"
    ) -> zuko.flows.Flow:

    filenames: List[str] = []
    with tempfile.TemporaryDirectory() as tempdir:

        optimizer = torch.optim.Adam(flow.parameters())
        q_i = flow 
        for i in range(num_iterations):
            q_bar = copy.deepcopy(q_i)

            with torch.no_grad():
                q_bar_samples, q_bar_log_prob = q_bar.forward().rsample_and_log_prob((num_q_bar_samples,))

                target_log_prob = target_log_prob_fn(q_bar_samples)
                reward = target_log_prob - q_bar_log_prob

            # TODO:
            """
            - RevESS
            - ELBO
            - WB Integration
            - Planar Reacher or Mixture of Gaussians target
            - Standalone demo in extra repo
            """
            # Optimize model using importance weights
            for _ in range(100): # inner loop
                dual: Callable[[float], float] = partial(dual_impl, epsilon=epsilon, reward=reward)
                eta_star = find_best_eta(dual)
                unnormalized_log_importance_weights = reward / (1 + eta_star)
                log_importance_weights = unnormalized_log_importance_weights - torch.logsumexp(unnormalized_log_importance_weights, 0)

                q_theta_log_prob = q_i.forward().log_prob(q_bar_samples)
                loss = -(log_importance_weights.exp() * q_theta_log_prob).sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Eval
            if i % eval_interval == 0 or i == num_iterations - 1:
                print(loss)
                
                axes: Tuple[Axes]
                fig, axes = plt.subplots(ncols=2)
                fig.set_size_inches(16, 8)
                plot_probability(axes[0], lambda x: target_log_prob_fn(x) / (1 + eta_star), f"target density (T={1 + eta_star:.2f})")
                plot_probability(axes[1], q_i.forward().log_prob, "q_theta")

                fname = os.path.join(tempdir, f"adaptive_smoothing_{i}.png")
                plt.savefig(fname)  # Save the figure
                filenames.append(fname)
                plt.show()
        
        images = [Image.open(f) for f in filenames]
        images[0].save(out_gif_path, save_all=True, append_images=images[1:], duration=200, loop=0)
    

def run(cfg):
    dim = 2

    target_log_prob_fn = U1
    flow = zuko.flows.NSF(dim)
    adaptive_smoothing_training(target_log_prob_fn, 100, 25, flow, 0.1, num_q_bar_samples=1000, out_gif_path="outputs/experiments/adaptive_smoothing.gif")