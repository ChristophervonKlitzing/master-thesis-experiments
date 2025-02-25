from .train import Trainer
from .flow import NormalizingFlow
from .exploration_model import GaussianExplorer
import zuko.flows as flows
import torch.optim as optim
import torch
from scipy.stats import special_ortho_group

torch.autograd.set_detect_anomaly(True)

def U1(z, exp=False):
    z_norm = torch.norm(z, 2, 1)
    add1 = 0.5 * ((z_norm - 2) / 0.4) ** 2
    add2 = -torch.log(
        torch.exp(-0.5 * ((z[:, 0] - 2) / 0.6) ** 2)
        + torch.exp(-0.5 * ((z[:, 0] + 2) / 0.6) ** 2)
        + 1e-9
    )
    f = -(add1 + add2)
    if exp:
        return torch.exp(f)
    else:
        return f
    



class BananaDistribution:
    def __init__(self, curvature, mean, var, translation):
        self.dim = len(mean)
        self.curvature = curvature

        self.base_dist = torch.distributions.MultivariateNormal(mean, var)
        self.translation = translation
        self.rotation = torch.tensor(special_ortho_group.rvs(self.dim), dtype=torch.float32)

    def sample(self, sample_shape):
        gaus_samples = self.base_dist.sample(sample_shape)
        x = torch.zeros_like(gaus_samples)

        # transform to banana shaped distribution
        x[:, 0] = gaus_samples[:, 0]
        x[:, 1] = gaus_samples[:, 1:] + self.curvature * torch.square(gaus_samples[:, 0].reshape(-1, 1)) - 1 * self.curvature
        # rotate samples
        x = torch.dot(x, self.rotation)

        # translate samples
        x = x + self.translation
        return x

    def log_prob(self, samples):
        # gaus_samples = torch.zeros_like(samples)

        # translate back
        samples = samples - self.translation
        # rotate back
        samples = samples @ self.rotation.T
        # transform back

        va = samples[:, 0]
        #print(gaus_samples[:, 1].shape)
        #print((samples[:, 1:] - self.curvature * torch.square(gaus_samples[:, 0].reshape(-1, 1)) + 1 * self.curvature).shape)
        vb = torch.squeeze(samples[:, 1:] - self.curvature * torch.square(va.reshape(-1, 1)) + 1 * self.curvature, dim=-1)
        gaus_samples = torch.stack([va, vb], dim=-1)
        # print(gaus_samples.shape, samples.shape)
        log_probs = self.base_dist.log_prob(gaus_samples)

        return log_probs



class BananaMixtureModel:
    def __init__(self, num_components, dim, log_Z=0., can_sample=True, sample_bounds=None) -> None:
        # parameters
        min_translation_val = -3
        max_translation_val = 3
        curvature_factor = 1.5

        self.num_components = num_components
        self.ndim = dim

        # set component distributions
        self.means = torch.zeros((self.num_components, self.ndim))
        self.covariances = torch.stack([torch.eye(self.ndim) * 0.1 for _ in range(self.num_components)])
        self.translations = torch.rand(size=(self.num_components, self.ndim)) * (max_translation_val - min_translation_val) + min_translation_val
        self.curvatures = torch.ones(self.num_components) * curvature_factor
        # set mixture weights

        self.mixture_weights = torch.ones(num_components) / num_components
        
        self.bananas: list[BananaDistribution] = []
        for i in range(self.num_components):
            self.bananas.append(BananaDistribution(self.curvatures[i], self.means[i], self.covariances[i],
                                                   self.translations[i]))

    def log_prob(self, samples: torch.Tensor) -> torch.Tensor:
        batched = samples.ndim == 2

        if not batched:
            samples = samples[None,]

        log_mixture_weights = torch.log(self.mixture_weights)
        banana_log_probs = torch.stack([banana.log_prob(samples) for banana in self.bananas])
        likelihoods = log_mixture_weights[:, torch.newaxis] + banana_log_probs

        # log sum exp trick for numerical stability
        result = torch.logsumexp(likelihoods, dim=0)

        if not batched:
            result = torch.squeeze(result, axis=0)
        return result



def run(cfg):
    torch.manual_seed(0)
    import random
    random.seed(3)
    import numpy as np
    np.random.seed(3)

    dim = 2

    # target_log_prob_fn = U1
    target_log_prob_fn = BananaMixtureModel(2, 2).log_prob

    trainer = Trainer(dim, target_log_prob_fn)
    trainer.train(1000, 20)
