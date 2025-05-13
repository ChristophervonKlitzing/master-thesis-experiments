from nflows.flows import Flow
import torch
from typing import Callable 
import bgflow as bg


def train_normal(
        generator: bg.BoltzmannGenerator, 
        num_iterations: int, 
        eval_interval: int, 
        num_samples_per_iteration: int = 100
    ) -> None:

    optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    for it in range(num_iterations):
        samples, model_energy = generator.sample(num_samples_per_iteration, with_energy=True)
        model_log_prob = -model_energy
        target_log_prob = -generator._target.energy(samples)
        loss = (model_log_prob - target_log_prob).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % eval_interval == 0 or it == num_iterations - 1:
            print(f"loss[{it}]: {loss.detach().item():.2f}")