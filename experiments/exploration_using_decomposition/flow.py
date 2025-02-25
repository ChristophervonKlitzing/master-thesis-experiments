from typing import Tuple
import torch
import torch.nn as nn
import zuko
import zuko.flows as flows


def _sum_rightmost(value: torch.Tensor, dim: int):
    r"""
    Sum out ``dim`` many rightmost dimensions of a given tensor.

    Args:
        value (Tensor): A tensor of ``.dim()`` at least ``dim``.
        dim (int): The number of rightmost dims to sum out.
    """
    if dim == 0:
        return value
    required_shape = value.shape[:-dim] + (-1,)
    return value.reshape(required_shape).sum(-1)


class NormalizingFlow(nn.Module):
    def __init__(self, flow: zuko.flows.Flow):
        super().__init__()
        self.flow = flow 
        
    def flow_inverse(self, x):
        f: zuko.distributions.NormalizingFlow = self.flow()
        z, ladj = f.transform.call_and_ladj(x)
        neg_log_det = _sum_rightmost(ladj, -1)
        return z, neg_log_det
    
    def flow_forward(self, z) -> tuple[torch.Tensor, torch.Tensor]:
        f: zuko.distributions.NormalizingFlow = self.flow()
        x, ladj = f.transform.inv.call_and_ladj(z)
        log_det = _sum_rightmost(ladj, -1)
        return x, log_det
    
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        f: zuko.distributions.NormalizingFlow = self.flow()
        return f.log_prob(x)
    
    def sample_and_log_prob(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        f: zuko.distributions.NormalizingFlow = self.flow()
        z = f.base.rsample((num_samples,))
        base_log_prob = f.base.log_prob(z)
        x, log_det = self.flow_forward(z)
        return x, base_log_prob - log_det