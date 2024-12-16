from ml_collections import ConfigDict
import numpyro.distributions as dist
import jax.numpy as jnp
import jax
from scipy.stats import special_ortho_group
import matplotlib.pyplot as plt
import chex
import haiku as hk
from approximate_inference_benchmark_refs.flows import ConfigurableFlow


class BananaTransform(ConfigurableFlow):
    """
    Note: This as a transform seems to be really unstable
    if the target distribution is more complex than the flow
    with this transform in it, can handle.
    E.g. additional modes 
    """
    def __init__(self, dim: int, key, initial_curvature: float = 0.0):
        super().__init__(ConfigDict())
        #self.dim = len(mean)
        #self.base_dist = dist.MultivariateNormal(mean, var)
        
        #np.random.seed(seed)
        #self.rotation = special_ortho_group.rvs(self.dim)
        initial_curvature_jnp = jnp.ones((1,)) * initial_curvature
        initial_curvature = hk.initializers.Constant(initial_curvature_jnp)

        initial_translation_jnp = jnp.zeros((dim,))
        initial_translation = hk.initializers.Constant(initial_translation_jnp)
        
        initial_rotation_jnp = jax.random.uniform(key)
        initial_rotation = hk.initializers.Constant(initial_rotation_jnp)

        self._rotation = hk.get_parameter(
            'rotation',
            shape=initial_rotation_jnp.shape,
            dtype=jnp.float32,
            init=initial_rotation
        )

        self._curvature = hk.get_parameter(
            'curvature',
            shape=initial_curvature_jnp.shape,
            dtype=jnp.float32,
            init=initial_curvature
        )

        """
        self._translation = hk.get_parameter(
            'translation',
            shape=initial_translation_jnp.shape,
            dtype=jnp.float32,
            init=initial_translation
        )
        """
    
    def _check_configuration(self, unused_config: ConfigDict):
        pass

    def get_actual_curvature(self):
        return self._curvature
    
    def transition_func(self, x: chex.Array):
        abs_x = jnp.abs(x)
        return jnp.where(abs_x <= 1, x * x, 2 * abs_x - 1) 
    
    def get_rotation_matrix(self, angle):
        # angle = jnp.sin(angle) * jnp.pi * 2
        return jnp.array(
            [[jnp.cos(angle), -jnp.sin(angle)],
             [jnp.sin(angle), jnp.cos(angle)]]
        )
    
    def rotate(self, x: chex.Array, R: chex.Array):
        return R @ x

    def transform_and_log_abs_det_jac(self, samples: chex.Array):
        batched = len(samples.shape) == 2
        if not batched:
            samples = jnp.expand_dims(samples, axis=0)
        
        R = self.get_rotation_matrix(self._rotation)
        samples = jax.vmap(self.rotate, in_axes=(0, None))(samples, R)

        log_det = jnp.zeros((samples.shape[0],))
        x = jnp.zeros_like(samples)

        # transform to banana shaped distribution
        x = x.at[:, 0].set(samples[:, 0])
        curv = self.get_actual_curvature()
        x = x.at[:, 1:].set(samples[:, 1:] +
                            curv * self.transition_func(samples[:, 0].reshape(-1, 1)) - 1 * self._curvature)

        x = x + 0.1

        """
        1 0 0 0 0
        curv 1 0 0 
        curv 0 1 0 
        """

        if not batched:
            x = jnp.squeeze(x, axis=0)
            log_det = jnp.squeeze(log_det, axis=0)

        return x, log_det
    
    def inv_transform_and_log_abs_det_jac(self, samples: chex.Array):
        batched = len(samples.shape) == 2
        if not batched:
            samples = jnp.expand_dims(samples, axis=0)

        log_det = jnp.zeros((samples.shape[0],))
        x = jnp.zeros_like(samples)

        samples = samples - 0.1

        x = x.at[:, 0].set(samples[:, 0])
        curv = self.get_actual_curvature()
        x = x.at[:, 1:].set(samples[:, 1:] -
                            curv * self.transition_func(x[:, 0].reshape(-1, 1)) + 1 * self._curvature)
        
        R = self.get_rotation_matrix(-self._rotation)
        samples = jax.vmap(self.rotate, in_axes=(0, None))(samples, R)

        if not batched:
            x = jnp.squeeze(x, axis=0)
            log_det = jnp.squeeze(log_det, axis=0)

        return x, log_det


class ReparametrizationTransform(ConfigurableFlow):
    def __init__(self, dim: int):
        super().__init__(ConfigDict())
        num_chol_params = (dim * dim + dim) // 2

        chol_init_vec = jnp.zeros((num_chol_params,))
        r = jnp.arange(dim)
        diagonal_indexes = (r + 1) * (r + 2) // 2 - 1
        chol_init_vec = chol_init_vec.at[diagonal_indexes].set(jnp.ones((dim,)) + 0.001 * jax.random.normal(jax.random.PRNGKey(0), (dim,)))
        
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

    def _get_chol_and_mu(self):
        chol = self._chol
        dim = self._mu.shape[0]
        L_indexes = jnp.tril_indices(dim)
        L = jnp.zeros((dim, dim))
        L = L.at[L_indexes].set(chol)
        log_det = jnp.log(jnp.abs(jnp.diagonal(L))).sum()
        return L, self._mu, log_det

    def _check_configuration(self, unused_config: ConfigDict):
        pass

    def transform_and_log_abs_det_jac(self, x):
        L, mu, log_det = self._get_chol_and_mu()
        return (L @ x.T).T + mu, log_det
    
    def inv_transform_and_log_abs_det_jac(self, x):
        L, mu, log_det = self._get_chol_and_mu()
        out = jnp.linalg.solve(L, (x - mu).T).T
        return out, -log_det
    
    