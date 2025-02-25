import torch


class DiagonalGaussian:
    """ A Gaussian mixture model with diagonal covariance matrices.

    Parameters:
        mean: torch.Tensor
            a one-dimensional tensor containing the component mean.

        cov: torch.Tensor
            a one-dimensional tensor containing the diagonal entries of the gaussian covariance.
    """

    def __init__(self, mean: torch.Tensor, cov: torch.Tensor):
        self.num_dimensions = mean.shape[0]
        self.mean = mean
        self.diagonal_chol_cov = torch.sqrt(cov) # diagonal entry vector
        
    @staticmethod
    def diagonal_gaussian_log_pdf(dim: int, mean: torch.Tensor, chol: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
        # samples.shape == (batch, dim)
        constant_part = - 0.5 * dim * torch.math.log(2 * torch.pi) - torch.sum(torch.log(chol))
        return constant_part - 0.5 * torch.sum(
            torch.square(torch.unsqueeze(1. / chol, 0) * (torch.unsqueeze(mean, 0) - samples)), dim=1)

    @staticmethod
    def diagonal_gaussian_entropy(chol: torch.Tensor):
        dim = chol.shape[0]
        return 0.5 * dim * (torch.math.log(2 * torch.pi) + 1) + torch.sum(torch.log(chol))
    
    @staticmethod
    def diagonal_gaussian_transform(base_samples: torch.Tensor, mu: torch.Tensor, chol: torch.Tensor):
        return torch.unsqueeze(mu, 0) + torch.unsqueeze(chol, 0) * base_samples

    @property
    def cov(self) -> torch.Tensor:
        return torch.square(self.diagonal_chol_cov)

    def entropy(self) -> torch.Tensor:
        return self.diagonal_gaussian_entropy(self.diagonal_chol_cov)

    def sample(self, num_samples: int) -> torch.Tensor:
        base_samples = torch.randn((num_samples, self.num_dimensions))
        return self.diagonal_gaussian_transform(base_samples, self.mean, self.diagonal_chol_cov)

    def log_density(self, samples: torch.Tensor) -> torch.Tensor:
        return self.diagonal_gaussian_log_pdf(self.num_dimensions, self.mean, self.diagonal_chol_cov, samples)
