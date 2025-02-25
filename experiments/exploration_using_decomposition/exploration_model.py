from typing import Tuple
import torch
import torch.nn as nn
import numpy as np

class GaussianExplorer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        num_chol_params = (dim * dim + dim) // 2

        chol_init_vec = torch.zeros((num_chol_params,), requires_grad=True)
        #r = torch.arange(dim)
        #diagonal_indexes = (r + 1) * (r + 2) // 2 - 1
        #chol_init_vec[diagonal_indexes] = torch.ones((dim,))
        
        self._chol = nn.Parameter(chol_init_vec)
        self._mu = nn.Parameter(torch.tensor([0.0, 0.0])) # nn.Parameter(torch.zeros((dim,)))

        self.initialized_freshly = False
    
    def _get_grads(self):
        chol = self._chol.grad
        dim = self._mu.shape[0]
        L_indexes = np.tril_indices(dim)
        diag_indexes = np.diag_indices(dim)
        L = torch.zeros((dim, dim))
        L[L_indexes] = chol # + 0.001 * torch.ones_like(chol)
        L[diag_indexes] = L[diag_indexes].exp()
        return self._mu.grad, L
    
    def _get_params(self):
        chol = self._chol
        dim = self._mu.shape[0]
        L_indexes = np.tril_indices(dim)
        diag_indexes = np.diag_indices(dim)
        L = torch.zeros((dim, dim))
        L[L_indexes] = chol # + 0.001 * torch.ones_like(chol)
        L[diag_indexes] = L[diag_indexes].exp()
        return self._mu, L
    
    def _get_pdf(self):
        mu, L = self._get_params()
        return torch.distributions.MultivariateNormal(mu, scale_tril=L)
    
    def sample_and_log_prob(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pdf = self._get_pdf()
        samples = pdf.rsample((num_samples,))

        chol = self._chol
        dim = self._mu.shape[0]
        L_indexes = np.tril_indices(dim)
        diag_indexes = np.diag_indices(dim)
        L = torch.zeros((dim, dim))
        L[L_indexes] = chol # + 0.001 * torch.ones_like(chol)

        log_det = L[diag_indexes].sum()
        return samples, pdf.log_prob(samples), log_det
    
    def sample(self, num_samples: int) -> torch.Tensor:
        pdf = self._get_pdf()
        return pdf.rsample((num_samples,))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        pdf = self._get_pdf()
        return pdf.log_prob(x)
    
    """
    def _cov_inverse(self):
        _, L = self._get_params()
        L_inv = torch.inverse(L)
        return L_inv @ L_inv.T

    def compute_natural_gradients(self, samples: torch.Tensor, grad_samples: torch.Tensor):
        # grad_samples.shape == (batch, dim)
        stepsize = 0.01
        cov_inv = self._cov_inverse()
        mu, L = self._get_params()
        N = samples.shape[0]
        
        outer = torch.bmm((samples - mu).unsqueeze(2), grad_samples.unsqueeze(1))
        batched_hessian = torch.bmm(cov_inv.unsqueeze(0).expand(N, -1, -1), outer)
        expected_hessian = batched_hessian.mean(0)
        grad_cov = 0.5 * expected_hessian
        grad_mu = grad_samples.mean(0)

        new_cov_inv = -2 * (-0.5 * cov_inv + stepsize * grad_cov)
        new_L = torch.inverse(torch.linalg.cholesky(new_cov_inv).mH.T)
        new_mu = (L.T @ L) @ (cov_inv @ mu + stepsize * (-2 * grad_cov @ mu + grad_mu))

        dim = self._mu.shape[0]
        diag_indexes = np.diag_indices(dim)
        new_chol = new_L
        new_chol[diag_indexes] = new_chol[diag_indexes].log()
        L_indexes = np.tril_indices(dim)
        new_chol = new_chol[L_indexes]

        #print(self._mu.shape, new_mu.shape)
        #print(self._chol.shape, new_chol.shape)
        with torch.no_grad():
            self._mu[:] = new_mu.detach()[:]
            self._chol[:] = new_chol.detach()[:]
    """

    def make_ngs(self):
        _, L = self._get_params()
        grad_mu, _ = self._get_grads()
        C = L.T @ L 
        self._mu.grad = C @ grad_mu
        self._chol.grad = self._chol.grad * 0.5
    

    def reset_search_state(self):
        self.initialized_freshly = True
        dim = self._mu.shape[0]
        num_chol_params = (dim * dim + dim) // 2

        chol_init_vec = torch.zeros((num_chol_params,), requires_grad=True)
        #r = torch.arange(dim)
        #diagonal_indexes = (r + 1) * (r + 2) // 2 - 1
        #chol_init_vec[diagonal_indexes] = torch.ones((dim,))
        
        self._chol[:] = chol_init_vec
        self._mu[:] = torch.tensor([0.0, 0.0]) # nn.Parameter(torch.zeros((dim,)))