import eacf
from eacf.setup_run.create_train_config import create_flow_config
from eacf.flow.build_flow import build_flow, FlowDistConfig
import eacf.utils
from eacf.utils.base import positional_dataset_only_to_full_graph, FullGraphSample
import eacf.utils.optimize
from omegaconf import DictConfig
import hydra


from .local_config import to_local_config

import jax
import jax.numpy as jnp
import optax

from .flow_wrapper import setup_flow_wrapper

import chex 
from typing import Callable, Tuple, Optional
from eacf.flow.aug_flow_dist import AugmentedFlow, FullGraphSample, AugmentedFlowParams, Positions, LogProb


def log_test_target(x: chex.Array):
    chex.assert_rank(x, 3)
    assert(x.shape[1] == 2)
    # (batch, n_nodes, dim)
    dist_vec = x[:, 0, :] - x[:, 1, :] # (batch, dim)
    distances = jnp.linalg.vector_norm(dist_vec, axis=1)
    return -jnp.square(distances - 2)


def run(args):
    """
    FullGraphSample:
        - positions: 
            - x: (batch, n_nodes, dim)
            - a: (batch, n_nodes, dim)
        - features: (batch, n_nodes, 1)
    """
    dim = 3

    # Manually initialize Hydra
    with hydra.initialize(config_path="config", version_base=None):
        # Compose the configuration (can apply overrides if needed)
        cfg = hydra.compose(config_name="lj13.yaml")
    
    key = jax.random.PRNGKey(0)

    print(cfg)
    cfg = to_local_config(cfg)
    cfg.flow.nodes = 2
    flow_config = create_flow_config(cfg)
    flow = build_flow(flow_config)

    num_nodes: int = cfg.flow.nodes


    full_graph_sample = positional_dataset_only_to_full_graph(jnp.zeros((1, num_nodes, dim)))
    flow_wrapper = setup_flow_wrapper(flow)
    flow_params = flow.init(key, full_graph_sample)

    # ========================================================
    

    opt_cfg = dict(cfg.training.optimizer)
    n_iter_warmup = 10
    n_iter_total = 200
    opt_cfg.pop('warmup_n_epoch')
    optimizer_config = eacf.utils.optimize.OptimizerConfig(
        **opt_cfg,
        n_iter_total=n_iter_total,
        n_iter_warmup=n_iter_warmup
    )
    optimizer, _ = eacf.utils.optimize.get_optimizer(optimizer_config)
    # optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(flow_params)

    @jax.jit
    def update(key, params, opt_state):
        def estimate_loss(flow_params: AugmentedFlowParams):
            return flow_wrapper.estimate_reverse_kl(flow_params, full_graph_sample.features[0], key, 200, log_test_target, cfg.training.aux_loss_weight)
        
        loss, grads = jax.value_and_grad(estimate_loss)(params)
        jax.debug.print("grad nan: {x}", x=jax.tree.reduce(lambda a, b: a + b, jax.tree.map(lambda x: jnp.isnan(x).sum(), grads)))
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return loss, params, opt_state

    print(flow_params.aux_target)
    print(flow_params.base)
    params = flow_params
    for step in range(200):
        key, subkey = jax.random.split(key)
        # Update parameters
        loss, params, opt_state = update(subkey, params, opt_state)

        # Logging
        # if step % 10 == 0:
        print(f"Step {step}, Loss: {loss:.4f}")
        

        if step % 20 == 0:
            # x_pos = jax.random.normal(key, (2, num_nodes, dim))
            x_pos = jnp.zeros((2, num_nodes, dim))
            x_pos = x_pos.at[:, 0, 0].set(2)
            test_samples_a = FullGraphSample(
                positions=x_pos,
                features=jnp.zeros((2, num_nodes, 1), dtype=int),
            )
            test_samples_b = FullGraphSample(
                positions=x_pos + 4,
                features=jnp.zeros((2, num_nodes, 1), dtype=int),
            )
            key, key_a, key_b = jax.random.split(key, 3)
            log_prob_a_test = flow_wrapper.estimate_marginal_log_prob(params, test_samples_a, key_a, 1000)
            log_prob_b_test = flow_wrapper.estimate_marginal_log_prob(params, test_samples_b, key_b, 1000)
            print(log_prob_a_test, log_prob_b_test)
            print(params.aux_target)
            print(params.base)
    
        print()
    
    