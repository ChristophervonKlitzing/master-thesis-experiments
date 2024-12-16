import haiku as hk
import jax.numpy as jnp
import numpyro.distributions as dist
import chex
import jax

from .model import Model
from approximate_inference_benchmark_refs.flows import SplineInverseAutoregressiveFlow, DiagonalAffine, AffineInverseAutoregressiveFlow
from .banana_transform import BananaTransform, ReparametrizationTransform
from ml_collections.config_dict import ConfigDict


class NormalizingFlow(Model):
    def __init__(self, dim: int, num_layers=2, hidden_size=8, name=None):
        """
        Normalizing Flow model composed of multiple affine coupling layers.
        
        Args:
        - num_layers: Number of coupling layers in the flow.
        - hidden_size: Number of hidden units in each coupling layer.
        """
        super().__init__(name=name)
        self.dim = dim
        self.base_dist = dist.MultivariateNormal(jnp.zeros((dim,)), jnp.eye(dim))

    def forward(self, x, reverse=False):
        cfg = ConfigDict()
        cfg.num_spline_bins = 10
        cfg.intermediate_hids_per_dim = 2
        cfg.num_layers = 1
        cfg.identity_init = True
        cfg.bias_last = False
        cfg.lower_lim = -10.
        cfg.upper_lim = 10.
        cfg.min_bin_size = 0.1
        cfg.min_derivative = -20.

        # flow: SplineInverseAutoregressiveFlow = SplineInverseAutoregressiveFlow(cfg)
        cfg2 = ConfigDict()
        cfg2.sample_shape = (self.dim,)
        flow_2: ReparametrizationTransform = ReparametrizationTransform(self.dim)
        # flow: BananaTransform = BananaTransform(self.dim, 1.0)

        cfg3 = ConfigDict()
        cfg3.intermediate_hids_per_dim = 64
        cfg3.num_layers = 4
        cfg3.identity_init = True
        cfg3.bias_last = False
        flow: AffineInverseAutoregressiveFlow = AffineInverseAutoregressiveFlow(cfg3)

        if reverse:
            def transform_fn(x: chex.Array):
                x, log_det1 = flow_2.inv_transform_and_log_abs_det_jac(x)
                x, log_det2 = flow.inv_transform_and_log_abs_det_jac(x)
                return x, log_det1 + log_det2
            
            # transform_fn = flow.inv_transform_and_log_abs_det_jac
        else:
            def transform_fn(x: chex.Array):
                x, log_det1 = flow.transform_and_log_abs_det_jac(x)
                x, log_det2 = flow_2.transform_and_log_abs_det_jac(x)
                return x, log_det2 + log_det1
            # transform_fn = flow.transform_and_log_abs_det_jac
        
        if len(x.shape) == 2:
            transform_fn = jax.vmap(transform_fn, in_axes=(0,), out_axes=(0, 0))
        
        x, log_det = transform_fn(x)

        return x, log_det
    
    def sample(self, key, num_samples):
        x = self.base_dist.sample(key, (num_samples,))
        x, log_det = self.forward(x)
        #print(log_det)
        #print("---_----")
        return x
    
    def log_prob(self, x):
        x, neg_log_det = self.forward(x, reverse=True)
        #print(neg_log_det)
        #print()
        return self.base_dist.log_prob(x) + neg_log_det

    
    