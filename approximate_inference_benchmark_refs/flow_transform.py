"""Code builds on algorithms.fab.flow.flow.py"""
from typing import NamedTuple, Callable, Tuple, Any

import chex
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

from .distrax_with_extra import Extra, BijectorWithExtra

Params = chex.ArrayTree
Update = Params
LogProb = chex.Array
LogDet = chex.Array
Sample = chex.Array


class FlowRecipe(NamedTuple):
    """Defines input needed to create an instance of the `Flow` callables."""
    make_bijector: Callable[[], BijectorWithExtra]
    n_layers: int
    config: Any
    dim: int
    compile_n_unroll: int = 2


class FlowTransformState(NamedTuple):
    bijector: Params


class FlowTransform(NamedTuple): 
    init: Callable[[chex.PRNGKey, Sample], FlowTransformState]
    forward_and_log_det: Callable[[FlowTransformState, Sample], Tuple[Sample, LogDet]]
    inverse_and_log_det: Callable[[FlowTransformState, Sample], Tuple[Sample, LogDet]]
    config: Any
    dim: int



class FlowForwardAndLogDet(nn.Module):
    bijector: BijectorWithExtra

    @nn.compact
    def __call__(self, x: chex.Array) -> Tuple[chex.Array, LogDet]:
        return self.bijector.forward_and_log_det(x)


class FlowInverseAndLogDet(nn.Module):
    bijector: BijectorWithExtra

    @nn.compact
    def __call__(self, y: chex.Array) -> Tuple[chex.Array, LogDet]:
        return self.bijector.inverse_and_log_det(y)




def create_flow_transform(recipe: FlowRecipe) -> FlowTransform:
    """Create a `FlowTransform` given the provided definition without a base-distribution"""

    bijector_block = recipe.make_bijector()
    forward_and_log_det_single = FlowForwardAndLogDet(bijector=bijector_block)
    inverse_and_log_det_single = FlowInverseAndLogDet(bijector=bijector_block)


    def forward_and_log_det(params: FlowTransformState, sample: Sample) -> Tuple[Sample, LogDet]:
        def scan_fn(carry, bijector_params):
            x, log_det_prev = carry
            y, log_det = forward_and_log_det_single.apply(bijector_params, x)
            chex.assert_equal_shape((log_det_prev, log_det))
            return (y, log_det_prev + log_det), None
        
        x = sample
        (y, log_det), _ = jax.lax.scan(scan_fn, init=(x, jnp.zeros(x.shape[:-1])), xs=params.bijector,
                                       unroll=recipe.compile_n_unroll)
        
        chex.assert_equal(x.shape[0], log_det.shape[0])  # Compare batch-dimensions
        chex.assert_equal_shape((y, x))
        
        return y, log_det
    

    def inverse_and_log_det(params: FlowTransformState, sample: Sample) -> Tuple[Sample, LogDet]:
        def scan_fn(carry, bijector_params):
            y, log_det_prev = carry
            x, log_det = inverse_and_log_det_single.apply(bijector_params, y)
            chex.assert_equal_shape((log_det_prev, log_det))
            return (x, log_det_prev + log_det), None

        y = sample
        log_prob_shape = y.shape[:-1]
        (x, log_det), _ = jax.lax.scan(scan_fn, init=(y, jnp.zeros(log_prob_shape)),
                                       xs=params.bijector, reverse=True,
                                       unroll=recipe.compile_n_unroll)
        
        chex.assert_equal_shape((x, y))
        chex.assert_equal(x.shape[0], log_det.shape[0]) # Compare batch-dimensions
        return x, log_det
    

    def init(key: chex.PRNGKey, sample: Sample) -> FlowTransformState:
        # Check shapes.
        chex.assert_tree_shape_suffix(sample, (recipe.dim,))

        params_bijector_single = inverse_and_log_det_single.init(key, sample)
        params_bijectors = jax.tree_map(lambda x: jnp.repeat(x[None, ...], recipe.n_layers, axis=0),
                                        params_bijector_single)
        return FlowTransformState(bijector=params_bijectors)
    

    bijector_block = FlowTransform(
        dim=recipe.dim,
        forward_and_log_det=forward_and_log_det,
        inverse_and_log_det=inverse_and_log_det,
        init=init,
        config=recipe.config
    )
    return bijector_block