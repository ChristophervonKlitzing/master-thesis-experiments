from functools import partial
import jax.flatten_util
import numpyro.distributions as dist
import jax.numpy as jnp
import chex
import jax
import haiku as hk
import optax
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple

from energy_potentials import U1, U2, U3
from experiments.benefits_of_ng_update.models.banana_transform import ReparametrizationTransform

from .models.model import ModelOp, create_model
from .models.gaussian_model import GaussianModel
from .models.flow_model import NormalizingFlow
from .util import batched_tree_ravel, tree_ravel, tree_unravel, plot_log_prob, wrap_log_prob_function, get_log_prob_num_calls

from .models.flow_transform import create_fab_flow, create_flow, Flow
from ml_collections.config_dict import ConfigDict




def create_objective_and_grad(dim, flow_1: Flow, flow_2: Flow, num_samples: int, target_log_prob_fn: Callable[[chex.Array], chex.Array], use_ng_1: bool, use_ng_2: bool):
    base_dist = dist.MultivariateNormal(jnp.zeros((dim,)), jnp.eye(dim))

    def model_log_prob_fn(flow_1_params: hk.MutableParams, flow_2_params: hk.MutableParams, transformed_sample: chex.Array):
        transformed_sample = jnp.expand_dims(transformed_sample, axis=0)
        x, neg_log_det_1 = flow_2.inverse_and_log_det(flow_2_params, transformed_sample)
        x, neg_log_det_2 = flow_1.inverse_and_log_det(flow_1_params, x)
        
        base_log_prob = base_dist.log_prob(x)
        model_log_prob = base_log_prob + neg_log_det_1 + neg_log_det_2
        return jnp.squeeze(model_log_prob)
    
    def estimate_FIM(log_prob_grad: chex.ArrayTree):
        # Vectorize gradient
        log_prob_grad_vec, _ = batched_tree_ravel(log_prob_grad)

        # Normalize FIM roughly to prevent exploding natural gradients (lr harder to tune)
        normalizer = jnp.abs(log_prob_grad_vec).mean()
        log_prob_grad_vec = log_prob_grad_vec / normalizer
        
        # Compute FIM
        def outer(a: chex.Array):
            return jnp.outer(a, a)
        FIM_estimates = jax.vmap(outer)(log_prob_grad_vec)
        FIM = FIM_estimates.mean(axis=0)
        
        return FIM
    
    def neg_kl(flow_1_params: hk.MutableParams, flow_2_params: hk.MutableParams, key: chex.PRNGKey):
        samples = base_dist.sample(key, (num_samples,))
        transformed_samples, log_det_1 = flow_1.forward_and_log_det(flow_1_params, samples)
        transformed_samples, log_det_2 = flow_2.forward_and_log_det(flow_2_params, transformed_samples)
        
        target_log_prob = target_log_prob_fn(transformed_samples)
        model_log_prob = base_dist.log_prob(samples) - log_det_1 - log_det_2

        _, log_prob_grad = jax.vmap(jax.value_and_grad(model_log_prob_fn, argnums=(0, 1)), in_axes=(None, None, 0), out_axes=(0, 0))(flow_1_params, flow_2_params, transformed_samples)
        
        if use_ng_1:
            log_prob_grad_flow_1: dict = log_prob_grad[0]
        if use_ng_2:
            log_prob_grad_flow_2: dict = log_prob_grad[1]

        if use_ng_1:
            FIM_flow_1 = estimate_FIM(log_prob_grad_flow_1)
        else:
            FIM_flow_1 = None
        
        if use_ng_2:
            FIM_flow_2 = estimate_FIM(log_prob_grad_flow_2)
        else:
            FIM_flow_2 = None
        
        return jnp.mean(target_log_prob - model_log_prob), (FIM_flow_1, FIM_flow_2)
    
    return jax.jit(jax.value_and_grad(neg_kl, has_aux=True, argnums=(0, 1)))
    # return jax.value_and_grad(neg_kl, has_aux=True, argnums=(0, 1))


def create_model_log_prob_fn(dim, flow_1: Flow, flow_2: Flow, flow_1_params: hk.MutableParams, flow_2_params: hk.MutableParams):
    base_dist = dist.MultivariateNormal(jnp.zeros((dim,)), jnp.eye(dim))
    def model_log_prob_fn(x: chex.Array):
        z = x
        z, neg_log_det_1 = flow_2.inverse_and_log_det(flow_2_params, z)
        z, neg_log_det_2 = flow_1.inverse_and_log_det(flow_1_params, z)
        log_prob = base_dist.log_prob(z) + neg_log_det_1 + neg_log_det_2
        log_prob = log_prob.at[jnp.isnan(log_prob)].set(-jnp.inf)
        return log_prob
    return model_log_prob_fn


def compute_elbo(dim, key: chex.PRNGKey, flow_1: Flow, flow_2: Flow, flow_1_params: hk.MutableParams, flow_2_params: hk.MutableParams, target_log_prob_fn: Callable[[chex.Array], chex.Array], num_samples: int = 500):
    base_dist = dist.MultivariateNormal(jnp.zeros((dim,)), jnp.eye(dim))

    samples = base_dist.sample(key, (num_samples,))
    base_log_prob = base_dist.log_prob(samples)

    transformed_samples, log_det_1 = flow_1.forward_and_log_det(flow_1_params, samples)
    transformed_samples, log_det_2 = flow_2.forward_and_log_det(flow_2_params, transformed_samples)
    
    target_log_prob = target_log_prob_fn(transformed_samples)
    model_log_prob = base_log_prob - log_det_1 - log_det_2

    return jnp.mean(target_log_prob - model_log_prob)


def update_params(params: hk.MutableParams, grad: hk.MutableParams, lr: float, FIM: Optional[chex.Array] = None) -> hk.MutableParams:
    if FIM is not None:
        params_vec, structure = tree_ravel(params)
        grad_vec, _ = tree_ravel(grad)
        natural_grad = jnp.linalg.solve(FIM, grad_vec)

        new_params_vec = params_vec + lr * natural_grad
        return tree_unravel(structure, new_params_vec)
    else:
        update = optax.tree_utils.tree_scalar_mul(lr, grad)
        return optax.tree_utils.tree_add(params, update)


def run(args):
    key = jax.random.PRNGKey(0)
    
    plot_on_log = True
    use_ng_1 = True
    use_ng_2 = True
    log_it = 125
    num_its = 1000

    lr = 0.01

    batch_size = 100
    dim = 2
    mu = jnp.array([-1., 1.])
    # mu = jnp.zeros_like(mu)
    correlation = 0.9
    cov = jnp.array(
        [[1., correlation],
         [correlation, 1.]]
    )

    t1 = dist.MultivariateNormal(mu, cov).log_prob
    t2 = lambda x: t1((x - mu) * (x - mu) + mu)
    raw_target_log_prob = t2
    counted_target_log_prob = wrap_log_prob_function(raw_target_log_prob)
    
    

    flow_cfg = ConfigDict()
    mode = 1

    if mode == 0:
        flow_cfg.flow_type = "AffineInverseAutoregressiveFlow"
        flow_cfg.intermediate_hids_per_dim = 64
        flow_cfg.num_layers = 2
        flow_cfg.identity_init = True
        flow_cfg.bias_last = False
    elif mode == 1:
        flow_cfg.flow_type = "SplineInverseAutoregressiveFlow"
        flow_cfg.num_spline_bins = 8
        flow_cfg.intermediate_hids_per_dim = 128
        flow_cfg.num_layers = 3
        flow_cfg.identity_init = True
        flow_cfg.bias_last = False
        flow_cfg.lower_lim = -5.
        flow_cfg.upper_lim = 5.
        flow_cfg.min_bin_size = 0.01
        flow_cfg.min_derivative = 1e-4
    elif mode == 2:
        flow_cfg.flow_type = "ConvAffineCouplingStack"
        flow_cfg.num_elem = dim
        flow_cfg.conv_kernel_shape=[(3, 3)]
        flow_cfg.conv_num_middle_layers=1
        flow_cfg.conv_num_middle_channels=1
        flow_cfg.is_torus=True
        flow_cfg.identity_init=True
    
    flow_1 = create_fab_flow(dim)
    flow_1_params = flow_1.init(key, jnp.zeros((dim,)))

    flow_2 = create_flow(key, ConfigDict(), ReparametrizationTransform, dim)
    flow_2_params = flow_2.init(key, jnp.zeros((dim,)))

    optimizer = optax.adam(learning_rate=1e-2)
    opt_state = optimizer.init(flow_1_params)

    objective_and_grad = create_objective_and_grad(dim, flow_1, flow_2, batch_size, counted_target_log_prob, use_ng_1, use_ng_2)
    for i in range(num_its):
        key, subkey_train, subkey_eval = jax.random.split(key, 3)
        (obj, FIM_tuple), (params_1_grad, params_2_grad) = objective_and_grad(flow_1_params, flow_2_params, subkey_train)
        FIM_tuple: Tuple[Optional[chex.Array], Optional[chex.Array]]
        FIM_flow_1, FIM_flow_2 = FIM_tuple

        # print(jnp.linalg.det(FIM_flow_1))
        if FIM_flow_1 is not None:
            FIM_flow_1 = FIM_flow_1 + jnp.eye(FIM_flow_1.shape[0])

        # print(params_2_grad)
        if jnp.isnan(obj):
            print("Objective:", obj)
            exit()

        # ========== Update Model Parameters ==========
        if not use_ng_1:
            params_1_grad = jax.tree_map(lambda x: -x, params_1_grad)
            updates, opt_state = optimizer.update(params_1_grad, opt_state, flow_1_params)
            flow_1_params = optax.apply_updates(flow_1_params, updates)
        else:
            flow_1_params = update_params(flow_1_params, params_1_grad, 0.05, FIM_flow_1)
        flow_2_params = update_params(flow_2_params, params_2_grad, lr, FIM_flow_2)


        if (i + 1) % log_it == 0 or i == 0:
            num_target_evals = get_log_prob_num_calls(counted_target_log_prob)
            elbo = compute_elbo(dim, subkey_eval, flow_1, flow_2, flow_1_params, flow_2_params, raw_target_log_prob)
            print(f"It: {i + 1}, ELBO: {elbo}, #Target-Evals: {num_target_evals}")

            if plot_on_log:
                # Plot the learned model and the target function
                model_log_prob = create_model_log_prob_fn(dim, flow_1, flow_2, flow_1_params, flow_2_params)

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))
                plot_log_prob(ax1, raw_target_log_prob, "target")
                plot_log_prob(ax2, model_log_prob, "model")
                fig.suptitle(f"Result after iteration {i + 1}")
                # fig.savefig(f"log/f_{i}.png")
                plt.show()
        
        
    # Plot the learned model and the target function
    model_log_prob = create_model_log_prob_fn(dim, flow_1, flow_2, flow_1_params, flow_2_params)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))
    plot_log_prob(ax1, raw_target_log_prob, "target")
    plot_log_prob(ax2, model_log_prob, "model")
    fig.suptitle(f"Final result after iteration {num_its}")
    plt.show()