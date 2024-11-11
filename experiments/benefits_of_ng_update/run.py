import jax.flatten_util
import numpyro.distributions as dist
import jax.numpy as jnp
import chex
import jax
import haiku as hk
import matplotlib.pyplot as plt
from typing import Callable

from .models.model import ModelOp, create_model
from .models.gaussian_model import GaussianModel
from .util import batched_tree_ravel, tree_ravel, tree_unravel, plot_log_prob









def create_objective_and_grad(model: ModelOp, num_samples: int, target_log_prob_fn: Callable[[chex.Array], chex.Array]):
    def model_log_prob_fn(model_params: hk.MutableParams, sample: chex.Array):
        return model.log_prob(model_params, sample)
    
    def neg_kl(model_params: hk.MutableParams, key: chex.PRNGKey):
        samples = model.sample(model_params, key, num_samples)
        target_log_prob = target_log_prob_fn(samples)
        model_log_prob, model_log_prob_grad = jax.vmap(jax.value_and_grad(model_log_prob_fn), in_axes=(None, 0), out_axes=(0, 0))(model_params, samples)
        
        # Vectorize gradient
        model_log_prob_grad_vec, _ = batched_tree_ravel(model_log_prob_grad)

        # Normalize FIM roughly to prevent exploding natural gradients (lr harder to tune)
        normalizer = jnp.abs(model_log_prob_grad_vec).mean()
        model_log_prob_grad_vec = model_log_prob_grad_vec / normalizer
        
        # Compute FIM
        def outer(a: chex.Array):
            return jnp.outer(a, a)
        FIM_estimates = jax.vmap(outer)(model_log_prob_grad_vec)
        FIM = FIM_estimates.mean(axis=0)
        
        return jnp.mean(target_log_prob - model_log_prob), FIM
    
    return jax.value_and_grad(neg_kl, has_aux=True)
    





def run(args):
    key = jax.random.PRNGKey(0)

    lr = 0.001
    dim = 2
    mu = jnp.array([-1., 1.])
    correlation = 0.999
    cov = jnp.array(
        [[1., correlation],
         [correlation, 1.]]
    )
    target_log_prob = dist.MultivariateNormal(mu, cov).log_prob

    model = create_model(GaussianModel, dim)
    model_params = model.init(key, jnp.zeros((dim,)))

    objective_and_grad = create_objective_and_grad(model, 100, target_log_prob)
    for i in range(100):
        key, subkey = jax.random.split(key)
        (obj, FIM), grad = objective_and_grad(model_params, subkey)
        
        if jnp.isnan(obj):
            print("Objective:", obj)
            print(FIM)
            print(jnp.linalg.det(FIM))

        if (i + 1) % 20 == 0 or i == 0:
            print(f"It: {i + 1}, ELBO: {obj}")
        
        raveled_model_params, params_structure = tree_ravel(model_params)
        gradient, _ = tree_ravel(grad)
        
        natural_gradient = jnp.linalg.solve(FIM, gradient)
        gradient = natural_gradient
        
        if jnp.isnan(obj):
            print(gradient)
            print(natural_gradient)
            print(raveled_model_params)
            break
        
        new_raveled_model_params = raveled_model_params + lr * gradient
        model_params = tree_unravel(params_structure, new_raveled_model_params)

    # Plot the learned model and the target function
    model_log_prob = lambda x: model.log_prob(model_params, x)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))
    plot_log_prob(ax1, target_log_prob, "target")
    plot_log_prob(ax2, model_log_prob, "model")
    plt.show()