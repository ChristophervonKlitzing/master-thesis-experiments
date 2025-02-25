import eacf
from eacf.setup_run.create_train_config import create_flow_config
from eacf.flow.build_flow import build_flow, FlowDistConfig
import eacf.utils
from eacf.utils.base import positional_dataset_only_to_full_graph, FullGraphSample
from eacf.train.max_lik_train_and_eval import eval_non_batched, calculate_forward_ess
import eacf.utils.optimize
from omegaconf import DictConfig
import hydra


from .local_config import to_local_config

import jax
import jax.numpy as jnp
import optax

from .flow_wrapper import FlowWrapper, setup_flow_wrapper

import chex 
from typing import Callable, Tuple, Optional
from eacf.flow.aug_flow_dist import AugmentedFlow, FullGraphSample, AugmentedFlowParams, Positions, LogProb
import distrax



def log_test_target(x: chex.Array):
    chex.assert_rank(x, 3)
    assert(x.shape[1] == 4)
    # (batch, n_nodes, dim)
    distances = jnp.zeros(x.shape[0])
    num_dists = 0 # could also be computed with gaussian sum formula
    for i in range(x.shape[1] - 1):
        for j in range(i + 1, x.shape[1]):
            distances = distances + jnp.linalg.vector_norm(x[:, i, :] - x[:, j, :], axis=1) # (batch, dim)
            num_dists += 1
    distances /= num_dists
    
    log_prob = jax.scipy.stats.multivariate_normal.logpdf(distances, 2.0, 1.0)
    return log_prob


def check_invariance(key: chex.PRNGKey, flow_wrapper: FlowWrapper, flow: AugmentedFlow, flow_params: AugmentedFlowParams, num_nodes: int, aux_loss_weight: float):
    joint_x_flow = flow.sample_apply(flow_params, jnp.zeros((num_nodes, 1), dtype=int), key, (5,))
    features, x_positions, a_positions = flow.joint_to_separate_samples(joint_x_flow)
    print(x_positions.shape)
    print("close to target: ", log_test_target(x_positions).mean())

    sample_a = x_positions[0]
    print("sample_a", sample_a)
    sample_b = sample_a[(1, 0, 2, 3), :]
    print("sample_b", sample_b)

    print("COM:", sample_a.mean(axis=0))
    test_samples_a = FullGraphSample(
        positions=sample_a,
        features=jnp.zeros((num_nodes, 1), dtype=int),
    )
    test_samples_b = FullGraphSample(
        positions=sample_b,
        features=jnp.zeros((num_nodes, 1), dtype=int),
    )
    key, key_a, key_b = jax.random.split(key, 3)
    log_prob_a_test = flow_wrapper.estimate_marginal_log_prob(flow_params, test_samples_a, key_a, 200)
    log_prob_b_test = flow_wrapper.estimate_marginal_log_prob(flow_params, test_samples_b, key_b, 200)
    
    print("log-probs sample a&b:")
    print("model:", log_prob_a_test, log_prob_b_test)
    print("target:", log_test_target(jnp.expand_dims(sample_a, 0)), log_test_target(jnp.expand_dims(sample_b, 0)))

    key, subkey = jax.random.split(key)
    info = eval_non_batched(flow_params, jnp.zeros((num_nodes, 1), dtype=int), subkey, flow, 2048, 64, log_test_target)
    print(info)

    batched_positions = x_positions[0:4]
    test_samples_c = FullGraphSample(
        positions=batched_positions,
        features=jnp.zeros((batched_positions.shape[0], num_nodes, 1), dtype=int),
    )

    #def batched_ml_loss
    ml_loss = flow_wrapper.estimate_ml_loss(key, flow_params, test_samples_c, aux_loss_weight)
    print("ML-loss:", ml_loss)



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
    
    # cfg = to_local_config(cfg)
    cfg.flow.nodes = 4
    flow_config = create_flow_config(cfg)
    flow = build_flow(flow_config)

    num_nodes: int = cfg.flow.nodes
    aux_loss_weight: float = cfg.training.aux_loss_weight

    full_graph_sample = positional_dataset_only_to_full_graph(jax.random.normal(key, (1, num_nodes, dim)))
    flow_wrapper = setup_flow_wrapper(flow)
    flow_params = flow.init(key, full_graph_sample)

    # ========================================================
    # check_invariance(key, flow_wrapper, flow, flow_params, num_nodes, aux_loss_weight)

    opt_cfg = dict(cfg.training.optimizer)
    n_iter_warmup = 10
    n_iter_total = 500
    opt_cfg.pop('warmup_n_epoch')
    optimizer_config = eacf.utils.optimize.OptimizerConfig(
        **opt_cfg,
        n_iter_total=n_iter_total,
        n_iter_warmup=n_iter_warmup
    )
    optimizer, _ = eacf.utils.optimize.get_optimizer(optimizer_config)
    # optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(flow_params)

    mean_sample = jnp.zeros((num_nodes, dim))
    mean_sample = mean_sample.at[1].set(jnp.array([4, 0, 0]))
    mean_sample = mean_sample.at[2].set(jnp.array([0, 4, 0]))
    mean_sample = mean_sample.at[3].set(jnp.array([0, 0, 4]))
    mean_sample = mean_sample.reshape(num_nodes * dim)

    reverse = False
    target_dist = distrax.MultivariateNormalDiag(mean_sample, 0.1 * jnp.ones(num_nodes * dim))
    def target_log_prob_reshaped(x: chex.Array):
        if len(x.shape) == 2:
            batched = False
            x = x.reshape((1, *x.shape))
        elif len(x.shape) == 3:
            batched = True 
        else:
            raise ValueError
        
        log_prob = target_dist.log_prob(x.reshape(x.shape[0], x.shape[1] * x.shape[2]))
        if not batched:
            log_prob = jnp.squeeze(log_prob)
        return log_prob

    @jax.jit
    def update(key, params, opt_state):
        def estimate_loss(flow_params: AugmentedFlowParams):
            if reverse:
                return flow_wrapper.estimate_reverse_kl(flow_params, full_graph_sample.features[0], key, 200, log_test_target, aux_loss_weight)
            else:
                samples: chex.Array = target_dist.sample(seed=key, sample_shape=200)
                samples = samples.reshape((samples.shape[0], num_nodes, dim))
                samples = samples - samples.mean(axis=1, keepdims=True)
                full_samples = positional_dataset_only_to_full_graph(samples)
                return flow_wrapper.estimate_ml_loss(key, flow_params, full_samples, aux_loss_weight)
        
        loss, grads = jax.value_and_grad(estimate_loss)(params)
        # jax.debug.print("grad nan: {x}", x=jax.tree.reduce(lambda a, b: a + b, jax.tree.map(lambda x: jnp.isnan(x).sum(), grads)))
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return loss, params, opt_state

    params = flow_params
    for step in range(n_iter_total):
        key, subkey = jax.random.split(key)
        # Update parameters
        loss, params, opt_state = update(subkey, params, opt_state)

        if step % 20 == 0 or step == n_iter_total - 1:
            key, subkey = jax.random.split(key)
            # check_invariance(subkey, flow_wrapper, flow, params, num_nodes, aux_loss_weight)
            # Logging
            # if step % 10 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")

            samples: chex.Array
            samples, target_log_prob = target_dist.sample_and_log_prob(seed=key, sample_shape=100)
            samples = samples.reshape((samples.shape[0], num_nodes, dim))
            samples = samples - samples.mean(axis=1, keepdims=True)
            full_samples = positional_dataset_only_to_full_graph(samples)
            model_log_prob = flow_wrapper.estimate_marginal_log_prob(params, full_samples, key, 100)
            model_log_prob_str = f"{model_log_prob.mean().item():.2f}±{model_log_prob.std().item():.2f}"
            target_log_prob_str = f"{target_log_prob.mean().item():.2f}±{target_log_prob.std().item():.2f}"
            print(f"Model-Log-Prob: {model_log_prob_str} | Target-Log-Prob: {target_log_prob_str}")

            #info = calculate_forward_ess(target_log_prob - model_log_prob, jnp.ones_like(target_log_prob))
            # info = eval_non_batched(params, jnp.zeros((num_nodes, 1), dtype=int), subkey, flow, 2048, 64, target_log_prob_reshaped)
            #print(info)

            print()
    
    