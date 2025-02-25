

from typing import NamedTuple, Callable, Optional, Tuple
import chex 
import jax 
import jax.numpy as jnp
from eacf.flow.aug_flow_dist import AugmentedFlow, AugmentedFlowParams, FullGraphSample, Positions, LogProb, GraphFeatures
from eacf.train.base import maybe_masked_mean
from eacf.utils.test import random_rotate_translate_permute


class FlowWrapper(NamedTuple):
    estimate_marginal_log_prob: Callable[[AugmentedFlowParams, FullGraphSample, chex.PRNGKey, int], chex.Array]
    estimate_reverse_kl: Callable[[AugmentedFlowParams, GraphFeatures, chex.PRNGKey, int, Callable[[chex.Array], chex.Array]], chex.Array]
    estimate_ml_loss: Callable[[chex.PRNGKey, AugmentedFlowParams, FullGraphSample, float], chex.Array]

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
        
        std = jnp.std(jnp.exp(log_w), axis=0)
        # jax.debug.print("std: {x}", x=std)
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
            log_p_x = target_log_prob(x_positions)
            log_p_a_given_x = flow.aux_target_log_prob_apply(params.aux_target,
                                        FullGraphSample(features=features, positions=x_positions), a_positions)
            
            return log_prob_flow, log_p_x, log_p_a_given_x, extra.aux_loss

        log_prob_flow, log_p_x, log_p_a_given_x, aux_loss_batch = forward(key)
        
        aux_loss = jnp.mean(aux_loss_batch)
        rev_kl_estimate = (log_prob_flow - log_p_x - log_p_a_given_x).mean()
        return rev_kl_estimate + aux_loss_weight * aux_loss
    
    def estimate_ml_loss(key: chex.PRNGKey, params: AugmentedFlowParams, x: FullGraphSample, aux_loss_weight: float, apply_random_rotation=True) -> chex.Array:
        if apply_random_rotation:
            key, subkey = jax.random.split(key)
            rotated_positions = random_rotate_translate_permute(x.positions, subkey, translate=True, permute=True)
            x = x._replace(positions=rotated_positions)

        aux_samples = flow.aux_target_sample_n_apply(params.aux_target, x, key)
        joint_samples = flow.separate_samples_to_joint(x.features, x.positions, aux_samples)

        #def batched_flow_log_prob_with_extra_apply(single_feature):
        #    jax.vmap(flow.log_prob_with_extra_apply, in_axes=(None, 0))(params, joint_samples)
        log_q, extra = flow.log_prob_with_extra_apply(params, joint_samples)
        mean_log_prob_q = jnp.mean(log_q)
        # Train by maximum likelihood.
        loss = -mean_log_prob_q
        aux_loss = jnp.mean(extra.aux_loss)
        loss = loss + aux_loss * aux_loss_weight
        return loss 


    return FlowWrapper(
        estimate_marginal_log_prob=estimate_marginal_log_prob,
        estimate_reverse_kl=estimate_reverse_kl,
        estimate_ml_loss=estimate_ml_loss,
    )
    
