from typing import NamedTuple, Callable, Optional, Tuple, Any, Type
import chex
import jax
import jax.numpy as jnp
import distrax
from flax import linen as nn
from approximate_inference_benchmark_refs import flows
import haiku as hk
from ml_collections.config_dict import ConfigDict

from approximate_inference_benchmark_refs.flow_transform import FlowTransformState
from experiments.benefits_of_ng_update.models.banana_transform import BananaTransform
from approximate_inference_benchmark_refs.build_flow_transform import FlowDistConfig, build_flow_transform


class Flow(NamedTuple): 
    init: Callable[[chex.PRNGKey, chex.Array], hk.MutableParams]
    forward_and_log_det: Callable[[hk.MutableParams, chex.Array], Tuple[chex.Array, chex.Array]]
    inverse_and_log_det: Callable[[hk.MutableParams, chex.Array], Tuple[chex.Array, chex.Array]]
    forward: Callable[[hk.MutableParams, chex.Array], chex.Array]
    inverse: Callable[[hk.MutableParams, chex.Array], chex.Array]

def create_fab_flow(dim: int):
    cfg = ConfigDict()

    cfg.base_loc = 0
    cfg.base_scale = 1

    cfg.n_layers = 4
    cfg.conditioner_mlp_units = [4, 4]  # Small MLP allows for fast local run, and can help stability.
    cfg.transform_type = "real_nvp"  # spline or real_nvp
    cfg.act_norm = False  # Set to true if using spline flow (especially if spline_max and spline_min are not known).
    cfg.identity_init = True
    cfg.spline_max = 10.  # If using spline then it helps to have bounds to roughly match problem.
    cfg.spline_min = -10.
    cfg.spline_num_bins = 8
    
    flow = build_flow_transform(FlowDistConfig(dim=dim, **cfg))

    def forward_and_log_det(flow_params: hk.MutableParams, z: chex.Array):
        return flow.forward_and_log_det(FlowTransformState(flow_params), z)
    
    def inverse_and_log_det(flow_params: hk.MutableParams, x: chex.Array):
        return flow.inverse_and_log_det(FlowTransformState(flow_params), x)
    
    def init(key: chex.PRNGKey, sample: chex.Array):
        return flow.init(key, sample).bijector
    
    return Flow(
        init=init,
        forward_and_log_det=forward_and_log_det,
        inverse_and_log_det=inverse_and_log_det,
        forward=lambda flow_params, z: forward_and_log_det(flow_params, z)[0],
        inverse=lambda flow_params, x: inverse_and_log_det(flow_params, x)[0],
    )

def create_flow(key, flow_cfg: ConfigDict, flow_type: Optional[Type[flows.ConfigurableFlow]], dim: int):
    use_banana = False
    def forward_and_log_det(x):
        if flow_type is not None:
            flow = flow_type(dim)
            return flow.transform_and_log_abs_det_jac(x)
        
        elif use_banana:
            full_log_det = jnp.array(0.)
            k = key
            for i in range(8):
                k, subkey = jax.random.split(k)
                flow: BananaTransform = BananaTransform(dim, subkey)
                if i % 2 == 1:
                    x = jnp.flip(x, axis=-1)
                x, log_det = flow.transform_and_log_abs_det_jac(x)
                if i % 2 == 1:
                    x = jnp.flip(x, axis=-1)
                full_log_det = full_log_det + log_det
                
            return x, full_log_det
        else:
            full_log_det = jnp.array(0.)
            for i in range(8):
                flow: flows.ConfigurableFlow = getattr(flows, flow_cfg.flow_type)(flow_cfg)
                if i % 2 == 1:
                    x = jnp.flip(x, axis=-1)
                x, log_det = flow.transform_and_log_abs_det_jac(x)
                if i % 2 == 1:
                    x = jnp.flip(x, axis=-1)
                full_log_det = full_log_det + log_det
                
            return x, full_log_det

    def inverse_and_log_det(x):
        if flow_type is not None:
            flow = flow_type(dim)
            return flow.inv_transform_and_log_abs_det_jac(x)
        elif use_banana:
            full_log_det = jnp.array(0.)
            k = key
            for i in range(4):
                k, subkey = jax.random.split(k)
                flow: BananaTransform = BananaTransform(dim, subkey)
                if i % 2 == 1:
                    x = jnp.flip(x, axis=-1)
                x, log_det = flow.inv_transform_and_log_abs_det_jac(x)
                if i % 2 == 1:
                    x = jnp.flip(x, axis=-1)
                full_log_det = full_log_det + log_det
                
            return x, full_log_det
        else:
            full_log_det = jnp.array(0.)
            for i in range(4):
                flow: flows.ConfigurableFlow = getattr(flows, flow_cfg.flow_type)(flow_cfg)
                if i % 2 == 1:
                    x = jnp.flip(x, axis=-1)
                x, log_det = flow.inv_transform_and_log_abs_det_jac(x)
                if i % 2 == 1:
                    x = jnp.flip(x, axis=-1)
                full_log_det = full_log_det + log_det
            return x, full_log_det

    flow_cfg.num_elem = dim
    flow_cfg.sample_shape = (dim,)

    flow_forward_fn = hk.without_apply_rng(hk.transform(forward_and_log_det))
    flow_inverse_fn = hk.without_apply_rng(hk.transform(inverse_and_log_det))

    flow_forward_fn_vec = jax.vmap(flow_forward_fn.apply, (None, 0), (0, 0))
    flow_inverse_fn_vec = jax.vmap(flow_inverse_fn.apply, (None, 0), (0, 0))

    def forward(flow_params: hk.MutableParams, z: chex.Array):
        x, _ = flow_forward_fn_vec(flow_params, z)
        return x
    
    def inverse(flow_params: hk.MutableParams, x: chex.Array):
        z, _ = flow_inverse_fn_vec(flow_params, x)
        return z
    
    def init(key: chex.PRNGKey, sample: chex.Array):
        return flow_forward_fn.init(key, sample)
    
    return Flow(
        init=init,
        forward_and_log_det=flow_forward_fn_vec,
        inverse_and_log_det=flow_inverse_fn_vec,
        forward=forward,
        inverse=inverse,
    )

