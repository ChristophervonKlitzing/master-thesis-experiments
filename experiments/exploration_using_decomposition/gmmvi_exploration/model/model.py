from abc import ABC, abstractmethod
import torch 
from torch import nn 
import zuko
from .diagonal_gaussian import DiagonalGaussian

class Model(ABC, nn.Module):
    @abstractmethod
    def log_density(self, samples: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def estimate_entropy(self, max_samples: int):
        """
        This returns an estimate of the models entropy. 
        max_samples must not be used completely if an estimate with fewer samples has reasonable accuracy
        or the entropy can be computed in closed-form.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_and_log_density(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError



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


class FlowModel(Model):
    def __init__(self, flow: zuko.flows.Flow):
        super().__init__()
        self.flow = flow 
        
    def flow_inverse(self, x):
        """
        Transforms samples from the target distribution into 
        samples from the latent-space/base distribution.
        """
        f: zuko.distributions.NormalizingFlow = self.flow()
        z, ladj = f.transform.call_and_ladj(x)
        neg_log_det = _sum_rightmost(ladj, -1)
        return z, neg_log_det
    
    def flow_forward(self, z) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Transforms latent-space samples (from base-distribution) into 
        samples from the target distribution.
        """
        f: zuko.distributions.NormalizingFlow = self.flow()
        x, ladj = f.transform.inv.call_and_ladj(z)
        log_det = _sum_rightmost(ladj, -1)
        return x, log_det
    
    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        f: zuko.distributions.NormalizingFlow = self.flow()
        return f.log_prob(x)
    
    def sample_and_log_density(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        f: zuko.distributions.NormalizingFlow = self.flow()
        z = f.base.rsample((num_samples,))
        base_log_prob = f.base.log_prob(z)
        x, log_det = self.flow_forward(z)
        return x, base_log_prob - log_det
    
    def estimate_entropy(self, max_samples):
        """
        Entropy can only be estimated for normalizing flows using samples. For now, this implementation uses the simplest
        definition of entropy (H(q) = -E_q(x)[log q(x)]). This estimate might get less noisy with the definition below, since
        the entropy of a simple gaussian base-distribution has an analytical solution.

        (b(z) is the latent-space/base distribution, q(x) is the flow distribution)
        H(q) = -E_q(x)[log q(x)] = -E_b(z)[log b(z) - log |det(...)|] = -E_b(z)[log b(z)] + E_b(z)[log |det(...)|] = H(b) + E_b(z)[log |det(...)|]
        """
        _, log_density = self.sample_and_log_density(max_samples)
        return -log_density.mean()




class DiagonalGaussianModel(Model):
    def __init__(self, mu: torch.Tensor, diag_cov: torch.Tensor):
        super().__init__()
        dim = mu.shape[0]
        self._diag_chol = nn.Parameter(diag_cov.sqrt())
        self._mu = nn.Parameter(mu)
        self._base = DiagonalGaussian(torch.zeros(dim), torch.ones(dim))

    def log_density(self, samples: torch.Tensor):
        dim = self._mu.shape[0]
        if (self._diag_chol<=0).any():
            print("chol is zero")
            print(self._diag_chol)
        return DiagonalGaussian.diagonal_gaussian_log_pdf(dim, self._mu, self._diag_chol, samples)

    def estimate_entropy(self, max_samples: int):
        return DiagonalGaussian.diagonal_gaussian_entropy(self._diag_chol)

    def sample_and_log_density(self, num_samples: int):
        base_samples = self._base.sample(num_samples)
        base_log_density = self._base.log_density(base_samples)
        transformed_samples = DiagonalGaussian.diagonal_gaussian_transform(base_samples, self._mu, self._diag_chol)
        log_det = self._diag_chol.abs().log().sum()
        return transformed_samples, base_log_density - log_det

