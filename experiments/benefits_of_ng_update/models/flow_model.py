import haiku as hk
import jax.numpy as jnp
import numpyro.distributions as dist
import chex

from .model import Model

# TODO: Implement a couple of normalizing flow types with few parameters 
# to check if they work significantly better with natural gradient updates or not.
class FlowModel(Model):
    def __init__(self, dim: int):
        super().__init__()
        num_chol_params = (dim * dim + dim) // 2

        chol_init_vec = jnp.zeros((num_chol_params,))
        r = jnp.arange(dim)
        diagonal_indexes = (r + 1) * (r + 2) // 2 - 1
        chol_init_vec = chol_init_vec.at[diagonal_indexes].set(jnp.ones((dim,)))

        chol_init = hk.initializers.Constant(chol_init_vec)
        mu_init = hk.initializers.Constant(jnp.zeros((dim,)))
        

        self._chol = hk.get_parameter(
            'chol',
            shape=[num_chol_params],
            dtype=jnp.float32,
            init=chol_init)
        self._mu = hk.get_parameter(
            'mu',
            shape=[dim],
            dtype=jnp.float32,
            init=mu_init)

    def _get_pdf(self):
        chol = self._chol
        dim = self._mu.shape[0]
        L_indexes = jnp.tril_indices(dim)
        L = jnp.zeros((dim, dim))
        L = L.at[L_indexes].set(chol)
        return dist.MultivariateNormal(self._mu, L @ L.T)
    
    def log_prob(self, x: chex.Array):
        return self._get_pdf().log_prob(x)
    
    def sample(self, key: chex.PRNGKey, num_samples: int) -> chex.Array:
        return self._get_pdf().sample(key, (num_samples,))
        

        