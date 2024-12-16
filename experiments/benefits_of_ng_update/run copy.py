from functools import partial
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
from .models.flow_model import NormalizingFlow
from .util import batched_tree_ravel, tree_ravel, tree_unravel, plot_log_prob, wrap_log_prob_function, get_log_prob_num_calls

from .models.flow_transform import create_flow, Flow
from ml_collections.config_dict import ConfigDict




def create_objective_and_grad(gaussian_model: ModelOp, flow_transform: Flow, num_samples: int, target_log_prob_fn: Callable[[chex.Array], chex.Array]):
    def model_log_prob_fn(model_params: hk.MutableParams, sample: chex.Array):
        return model.log_prob(model_params, sample)
    
    def neg_kl(model_params: hk.MutableParams, key: chex.PRNGKey):
        samples = model.sample(model_params, key, num_samples)
        
        target_log_prob = target_log_prob_fn(samples)
        model_log_prob, model_log_prob_grad = jax.vmap(jax.value_and_grad(model_log_prob_fn), in_axes=(None, 0), out_axes=(0, 0))(model_params, samples)
        model_log_prob_grad: dict

        # Extract only gradients for reparametrization transform for NG update
        reparam_key = 'normalizing_flow/~forward/reparametrization_transform'
        model_log_prob_grad = model_log_prob_grad.pop(reparam_key)

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
    
    plot_on_log = True
    use_natural_gradient = True
    log_it = 100
    num_its = 1000

    lr = 0.001
    batch_size = 100
    dim = 2
    mu = jnp.array([-1., 1.])
    correlation = 0.99
    cov = jnp.array(
        [[1., correlation],
         [correlation, 1.]]
    )

    t1 = dist.MultivariateNormal(mu, cov).log_prob
    raw_target_log_prob = lambda x: t1((x - mu) * (x - mu) + mu)
    counted_target_log_prob = wrap_log_prob_function(raw_target_log_prob)
    
    gaussian_model = create_model(GaussianModel, dim)
    gaussian_params = gaussian_model.init(key, jnp.zeros((dim,)))

    flow_cfg = ConfigDict()
    flow_cfg.flow_type = "AffineInverseAutoregressiveFlow"
    flow_cfg.intermediate_hids_per_dim = 2
    flow_cfg.num_layers = 2
    flow_cfg.identity_init = True
    flow_cfg.bias_last = False
    
    flow_transform = create_flow(flow_cfg, dim)
    flow_params = flow_transform.init(key, jnp.zeros((dim,)))
    #model = create_model(NormalizingFlow, dim)
    #model_params = model.init(key, jnp.zeros((dim,)))

    objective_and_grad = create_objective_and_grad(gaussian_model, flow_transform, batch_size, counted_target_log_prob)
    for i in range(num_its):
        key, subkey = jax.random.split(key)
        (obj, FIM), grad = objective_and_grad(model_params, subkey)
        FIM: chex.Array

        if jnp.isnan(obj):
            print("Objective:", obj)
            print(FIM)
        
        raveled_model_params, params_structure = tree_ravel(model_params)
        raw_gradient, _ = tree_ravel(grad)
        
        if not use_natural_gradient:
            gradient = raw_gradient
        
        if use_natural_gradient:
            """
            indexes = jnp.nonzero(1 - zero_mask)[0]
            # indexes = jnp.arange(FIM.shape[0])

            FIM_reduced = FIM[indexes, :].T[indexes, :].T
            raw_gradient_reduced = raw_gradient[indexes]
            print(raw_gradient_reduced)

            gradient = jnp.zeros_like(raw_gradient)
            # print(indexes.shape)
            
            if jnp.abs(jnp.linalg.det(FIM_reduced)) <= 0.000001:
                print("Make FIM invertible through a hacky trick")
                print(jnp.linalg.det(FIM_reduced))
                FIM_reduced = 0.1 * jnp.eye(FIM_reduced.shape[0]) + FIM_reduced
                
                # print(FIM_reduced)

            gradient = gradient.at[indexes].set(jnp.linalg.solve(FIM_reduced, raw_gradient_reduced))
            
            """

            if jnp.abs(jnp.linalg.det(FIM)) <= 0.00000001:
                print("Make FIM invertible through a hacky trick")
                FIM = 0.01 * jnp.eye(FIM.shape[0]) + FIM
            else:
                print("Use NG update")
            
            gradient = raw_gradient.at[-5:].set(jnp.linalg.solve(FIM, raw_gradient[-5:]))

            # gradient = gradient.at[:-5].set(gradient[:-5] * 2)
            # gradient = gradient.at[:-5].set(raw_gradient[:-5] * 10)
     
        
        if jnp.isnan(obj):
            print(raw_gradient)
            print(gradient)
            print(raveled_model_params)
            break
        
        new_raveled_model_params = raveled_model_params + lr * gradient
        if not jnp.isnan(raveled_model_params).any() and jnp.isnan(new_raveled_model_params).any():
            print("Gradient Update introduced nan")
            print(FIM)
            print(gradient)
            print(raveled_model_params)
            print()
        model_params = tree_unravel(params_structure, new_raveled_model_params)

        if (i + 1) % log_it == 0 or i == 0:
            num_target_evals = get_log_prob_num_calls(counted_target_log_prob)
            print(f"It: {i + 1}, ELBO: {obj}, #Target-Evals: {num_target_evals}")

            if plot_on_log:
                # Plot the learned model and the target function
                model_log_prob = lambda x: model.log_prob(model_params, x)

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))
                plot_log_prob(ax1, raw_target_log_prob, "target")
                plot_log_prob(ax2, model_log_prob, "model")
                fig.suptitle(f"Result after iteration {i + 1}")
                plt.show()

    # Plot the learned model and the target function
    model_log_prob = lambda x: model.log_prob(model_params, x)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))
    plot_log_prob(ax1, raw_target_log_prob, "target")
    plot_log_prob(ax2, model_log_prob, "model")
    fig.suptitle(f"Final result after iteration {num_its}")
    plt.show()