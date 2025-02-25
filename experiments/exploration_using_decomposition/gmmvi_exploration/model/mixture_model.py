from typing import Callable
import torch
from torch import nn 
import copy

from .model import Model, DiagonalGaussianModel


class MixtureModel:
    def __init__(self, num_dimensions: int, max_samples_for_entropy: int = 50):
        self.num_dimensions = num_dimensions
        self.max_samples_for_entropy = max_samples_for_entropy
        # self.component_model_generator = component_model_generator

        self.component_models: list[Model] = []
        self.component_log_weights: torch.Tensor = torch.zeros((0,)) # = torch.tensor([1 / num_initial_components] * num_initial_components).log()

    def get_num_components(self) -> int:
        return len(self.component_models)
    
    def log_density(self, samples: torch.Tensor) -> torch.Tensor:
        """Returns a tensor of shape (#samples,)"""
        component_log_densities = self.component_log_densities(samples)
        return (component_log_densities + self.component_log_weights.unsqueeze(1)).logsumexp(0)
    
    def component_log_densities(self, samples: torch.Tensor):
        """Returns a tensor of shape (num_components, #samples)"""
        return self._component_log_densities(self.component_models, samples)
    
    def component_sample_and_log_densities(self, num_samples_per_component: int):
        component_samples_and_log_densities = [model.sample_and_log_density(num_samples_per_component) for model in self.component_models]
        component_samples = [samples_and_log_prob[0] for samples_and_log_prob in component_samples_and_log_densities]
        component_log_densities = [samples_and_log_prob[1] for samples_and_log_prob in component_samples_and_log_densities]
        return component_samples, component_log_densities

    @staticmethod
    def _component_log_densities(component_models: list[Model], samples: torch.Tensor):
        return torch.stack([component_model.log_density(samples) for component_model in component_models])

    def get_average_entropy(self) -> torch.Tensor:
        component_entropies = self.component_entropies()
        return (component_entropies * self.component_log_weights.exp()).sum()
    
    def component_entropies(self):
        return torch.stack([component_model.estimate_entropy(self.max_samples_for_entropy) for component_model in self.component_models])

    
    
    def _compute_component_log_responsibilities(self, index: int, samples: torch.Tensor, component_models: list[Model]):
        """
        (x = samples, o = index)
        q(o|x) = q(x|o)q(o) / \sum_o{q(x|o)q(o)}
        Returns log responsibilities log q(o|x) for component o of shape (#samples,)
        """
        weighted_log_densities = self._component_log_densities(component_models, samples) + self.component_log_weights.unsqueeze(1)
        if torch.isnan(weighted_log_densities).any():
            print(weighted_log_densities)
        weighted_component_log_density = weighted_log_densities[index] # shape (#samples,)
        log_density = weighted_log_densities.logsumexp(0) # shape (#samples,)
        return weighted_component_log_density - log_density
    
    def component_log_responsibilities(self, component_samples: list[torch.Tensor]):
        """
        Computes the log-responsibilities for the given samples component-wise.
        The parameters of the models receive no gradient-update but the models are still end-to-end differentiable.
        This is accomplished by making a copy of each component-model. Due to this operation being expensive for large models,
        it is only done once for all component log-responsibilities. This current state of the models
        is used in this function. This is equivalent to making the lower bound tight (E-step).

        The input and return parameter are both of type list in case some components get less samples than other!
        Lists are in this case just more flexible than tensors.

        Parameters:
            - component_samples: 
                List of component-samples -> len(component_samples) == num_components.
        Returns:
            A list of log-responsibilities [q(o|x) for o, x in enumerate(component_samples)] of length num_components
        """
        copied_component_models = copy.deepcopy(self.component_models)
        return [self._compute_component_log_responsibilities(i, samples, copied_component_models) for i, samples in enumerate(component_samples)]

    def train(self):
        # Put model into training mode
        for model in self.component_models:
            model.train()
    
    def eval(self):
        # Put model into eval mode
        for model in self.component_models:
            model.eval()