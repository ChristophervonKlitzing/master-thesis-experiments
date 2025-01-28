

from typing import NamedTuple, Callable, Optional, Tuple
import chex 
import jax 
import jax.numpy as jnp
from eacf.flow.aug_flow_dist import AugmentedFlow, AugmentedFlowParams, FullGraphSample, Positions, LogProb, GraphFeatures
from eacf.train.base import maybe_masked_mean

class FlowWrapper(NamedTuple):
    estimate_marginal_log_prob: Callable[[AugmentedFlowParams, FullGraphSample, chex.PRNGKey, int, Optional[chex.Array]], chex.Array]
    estimate_reverse_kl: Callable[[AugmentedFlowParams, GraphFeatures, chex.PRNGKey, int, Callable[[chex.Array], chex.Array]], chex.Array]

def setup_flow_wrapper(flow: AugmentedFlow):
    def estimate_marginal_log_prob(
            params: AugmentedFlowParams, 
            sample: FullGraphSample, 
            key: chex.PRNGKey,
            K: int,):
        # Estimates log{q(x)} = log{int q(x, a) da} where a is the augmented variable vector
        key, subkey = jax.random.split(key)
        x_augmented, log_p_a = flow.aux_target_sample_n_and_log_prob_apply(params.aux_target, sample, subkey, K)
        sample = jax.tree_map(lambda x: jnp.repeat(x[None, ...], K, axis=0), sample)
        # features, x_pos, x_a = flow.joint_to_separate_samples(sample)
        # print(features.shape, x_pos.shape, x_augmented.shape)
        joint_sample = flow.separate_samples_to_joint(sample.features, sample.positions, x_augmented)
        log_q = jax.vmap(flow.log_prob_apply, in_axes=(None, 0))(params, joint_sample)
        chex.assert_equal_shape((log_p_a, log_q))
        log_w = log_q - log_p_a

        marginal_log_lik = jax.nn.logsumexp(log_w, axis=0) - jnp.log(jnp.array(K))
        return marginal_log_lik
    
    def estimate_reverse_kl(
            params: AugmentedFlowParams, 
            single_feature: GraphFeatures,
            key: chex.PRNGKey, 
            n_samples: int,
            target_log_prob: Callable[[chex.Array], chex.Array],
            aux_loss_weight: float,
        ):

        def forward(key: chex.PRNGKey):
            joint_x_flow, log_prob_flow, extra = flow.sample_and_log_prob_with_extra_apply(params, single_feature, key, (n_samples,))
            features, x_positions, a_positions = flow.joint_to_separate_samples(joint_x_flow)
            jax.debug.print("x_pos: {x} a_pos: {y}", x=jnp.isnan(x_positions).sum(), y=jnp.isnan(a_positions).sum())
            log_p_x = target_log_prob(x_positions)
            log_p_a_given_x = flow.aux_target_log_prob_apply(params.aux_target,
                                        FullGraphSample(features=features, positions=x_positions), a_positions)
            
            return log_prob_flow, log_p_x, log_p_a_given_x, extra.aux_loss

        log_prob_flow, log_p_x, log_p_a_given_x, aux_loss_batch = forward(key)

        jax.debug.print("target log-prob: {x}", x=log_p_x.mean())
        
        # Calculate reverse KL.
        jax.debug.print("log_prob_flow #nan: {x}", x=jnp.isnan(log_prob_flow).sum())
        jax.debug.print("log_p_x #nan: {x}", x=jnp.isnan(log_p_x).sum())
        jax.debug.print("log_p_a_given_x #nan: {x}", x=jnp.isnan(log_p_a_given_x).sum())

        aux_loss = jnp.mean(aux_loss_batch)
        rev_kl_estimate = (log_prob_flow - log_p_x - log_p_a_given_x).mean()
        return rev_kl_estimate + 10.0 * aux_loss

    return FlowWrapper(
        estimate_marginal_log_prob=estimate_marginal_log_prob,
        estimate_reverse_kl=estimate_reverse_kl,
    )
    
