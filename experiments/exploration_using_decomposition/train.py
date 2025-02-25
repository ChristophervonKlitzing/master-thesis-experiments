import copy
import math
from typing import Callable, Dict, Iterator, Tuple
import torch
from torch import nn 
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import zuko.flows as flows
from matplotlib.axes import Axes
from .flow import NormalizingFlow
from .exploration_model import GaussianExplorer


from .gmmvi_exploration.gmmvi_modules.component_adaptation import VipsComponentAdaptation
from .gmmvi_exploration.model.mixture_model import MixtureModel
from .gmmvi_exploration.model.model import FlowModel, DiagonalGaussianModel
from .gmmvi_exploration.sample_db import SampleDB


def _color_generator(num_colors: int, cmap_name='viridis'):
    cmap = get_cmap(cmap_name, num_colors)
    colors = [cmap(i / num_colors) for i in range(num_colors)]
    return colors

class Trainer:
    def __init__(self, dim: int, target_lnpdf: Callable[[torch.Tensor], torch.Tensor]):
        self.num_dimensions = dim 
        self.num_samples_per_component = 100
        self.target_lnpdf = target_lnpdf
        self.optimizers: list[torch.optim.Optimizer] = []

        num_initial_components = 1

        self.max_reward_history_length = 100
        self.reward_history = torch.finfo(torch.float32).min * torch.ones((num_initial_components, self.max_reward_history_length))
        self.weight_history = torch.finfo(torch.float32).min * torch.ones((num_initial_components, self.max_reward_history_length))

        threshold_for_add_adaption = [ 5000., 1000.0, 500.0, 200.0, 100.0, 50.0 ]

        self.flow_generator = lambda: FlowModel(flows.NSF(dim, transforms=3, hidden_features=[64] * 3))
        
        self.sample_db = SampleDB(dim, keep_samples=True, max_samples=1000)
        self.mixture_model = MixtureModel(dim)
        self.component_adapter = VipsComponentAdaptation(
            self,
            self.mixture_model, self.sample_db, self.target_lnpdf,
            prior_mean=0.0,
            initial_cov=5.0,
            del_iters=100,
            add_iters=40,
            max_components=2,
            min_weight_for_del_heuristic=1e-6,
            num_database_samples=0,
            num_prior_samples=1000,
            thresholds_for_add_heuristic=threshold_for_add_adaption,
        )

        for _ in range(num_initial_components):
            component_model = self.flow_generator()
            self.mixture_model.component_models.append(component_model)
            self.optimizers.append(torch.optim.AdamW(component_model.parameters(), lr=0.01))
        self.mixture_model.component_log_weights = torch.tensor([1 / num_initial_components] * num_initial_components).log()

        self.component_one_count = None
        self.gaussian_one = None

    def add_component(self, init_weight: float, new_mean: torch.Tensor, new_cov: torch.Tensor, threshold_for_add: torch.Tensor, des_entropy: torch.Tensor):
        """
        new_mean.shape == (self.num_dimensions,)
        new_cov.shape == (self.num_dimensions,)
        threshold_for_add.shape == (1,)
        des_entropy.shape == (1,)
        """
        # TODO: Instead pre-train a gaussian here or pre-train the model to the given new_mean and new_cov 
        # new_component = self.component_model_generator()
        new_component = DiagonalGaussianModel(new_mean, new_cov)
        self.component_one_count = 0
        self.mixture_model.component_models.append(new_component)
        init_weight_tensor = torch.tensor([init_weight])
        self.mixture_model.component_log_weights = torch.concat((self.mixture_model.component_log_weights, init_weight_tensor.log()))
        
        self.optimizers.append(torch.optim.AdamW(new_component.parameters(), lr=0.01))

        self.reward_history = torch.vstack((
            self.reward_history, torch.ones((1, self.max_reward_history_length)) * torch.finfo(torch.float32).min
        ))
        self.weight_history = torch.vstack((
            self.weight_history, torch.ones((1, self.max_reward_history_length)) * init_weight
        ))
        
    def remove_component(self, index: int):
        def _pop_row(arr: torch.Tensor):
            return torch.vstack((arr[:index], arr[index + 1:]))
        self.mixture_model.component_models.pop(index)
        self.mixture_model.component_log_weights = _pop_row(self.mixture_model.component_log_weights)
        
        self.optimizers.pop(index)

        self.reward_history = _pop_row(self.reward_history)
        self.weight_history = _pop_row(self.weight_history)
    
    def compute_component_loss_and_reward(self, log_target_densities: torch.Tensor, log_responsibilities: torch.Tensor, log_model_densities: torch.Tensor):
        reward = (log_target_densities + log_responsibilities - log_model_densities).mean()
        loss = -reward
        return loss, reward.detach()
    
    def compute_elbo(self):
        ...
    
    def train_step(self, iteration: int):
        if self.component_one_count is not None and self.component_one_count is not None:
            self.component_one_count += 1 

        if self.component_one_count is not None and self.component_one_count == 21:
            self.gaussian_one = self.mixture_model.component_models[1]
            self.mixture_model.component_models[1] = self.flow_generator()
            self.optimizers[1] = torch.optim.AdamW(self.mixture_model.component_models[1].parameters(), lr=0.01)
            
        component_samples, component_log_model_densities = self.mixture_model.component_sample_and_log_densities(self.num_samples_per_component)
        component_log_target_densities = [self.target_lnpdf(samples) for samples in component_samples]
        component_log_q = self.mixture_model.component_log_responsibilities(component_samples)

        if self.component_one_count is not None and self.component_one_count >= 21 and self.component_one_count <= 30:
            component_log_target_densities[1] = self.gaussian_one.log_density(component_samples[1])
        
        for samples, log_target_lnpdfs in zip(component_samples, component_log_target_densities):
            self.sample_db.add_samples(samples, log_target_lnpdfs)

        arg_list =  zip(component_log_target_densities, component_log_q, component_log_model_densities)
        component_losses_and_rewards = [
            self.compute_component_loss_and_reward(
                log_target_densities, 
                log_responsibilities, 
                log_model_densities
            )
            for log_target_densities, log_responsibilities, log_model_densities in arg_list
        ]
        component_losses = [loss_and_reward[0] for loss_and_reward in component_losses_and_rewards]
        component_rewards = torch.stack([loss_and_reward[1] for loss_and_reward in component_losses_and_rewards]) # (#components,)
        average_reward = (self.mixture_model.component_log_weights.exp() * component_rewards).sum()

        # Update the components
        for loss, optimizer in zip(component_losses, self.optimizers):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Update the mixture weights
        new_log_weights = component_rewards - component_rewards.logsumexp(0)
        new_log_weights = torch.clip(new_log_weights, torch.tensor(-55.0))
        self.mixture_model.component_log_weights = new_log_weights

        # Adapt number of components in mixture model if necessary
        self.component_adapter.adapt_number_of_components(iteration)

        print(self.mixture_model.component_log_weights.exp().detach().numpy())
        return {"average-return": average_reward.detach().item()}

    def train(self, num_iters: int, eval_interval: int):
        self.mixture_model.train()
        for iteration in range(num_iters):
            metrics = self.train_step(iteration)
            if (iteration % eval_interval == 0) or (iteration == num_iters - 1):
                self.mixture_model.eval()
                metric_representation = ", ".join(f"{key}: {value:.4f}" for key, value in metrics.items())
                print(f"It {iteration}: {metric_representation}")
                self.visualize()
                self.mixture_model.train()

    def visualize(self):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.5, 4.5))
        ax1: Axes
        ax2: Axes
        ax3: Axes

        x = torch.linspace(-5, 5, 100)
        y = torch.linspace(-5, 5, 100)
        X, Y = torch.meshgrid(x, y)
        grid_points = torch.stack((X.ravel(), Y.ravel()), axis=-1)

        scalar_values = self.target_lnpdf(grid_points).reshape(X.shape).exp() + 1e-6
        ax1.contourf(X.detach().numpy(), Y.detach().numpy(), scalar_values.detach().numpy(), levels=100, cmap='viridis')
        ax1.set_title("target log-density")

        colors = _color_generator(num_colors=self.mixture_model.get_num_components())
        component_densities = self.mixture_model.component_log_densities(grid_points)
        for i in range(self.mixture_model.get_num_components()):
            scalar_values = component_densities[i].reshape(X.shape).detach().exp() + 1e-6
            ax2.contour(X.detach().numpy(), Y.detach().numpy(), scalar_values.detach().numpy(), levels=10, colors=[colors[i]])
        ax2.set_title("mixture-model components")
        
        scalar_values = self.mixture_model.log_density(grid_points).reshape(X.shape).exp() + 1e-6
        ax3.contourf(X.detach().numpy(), Y.detach().numpy(), scalar_values.detach().numpy(), levels=100, cmap='viridis')
        ax3.set_title("mixture-model log-density")
        plt.show()

