import jax.flatten_util
import jax.numpy as jnp
import chex
import jax
import optax
from matplotlib.axes import Axes
from typing import Any, Callable, List, NamedTuple, Tuple
from functools import reduce
from operator import mul


class TreeStructure(NamedTuple):
    pytree_structure: Any
    leaf_shapes: List[Tuple[int, ...]]


def tree_ravel(pytree: dict):
    flattened_pytree, structure = jax.tree_flatten(pytree)
    flattened_pytree: List[chex.Array]
    leaf_shapes = [x.shape for x in flattened_pytree]
    flattened_pytree = [jnp.ravel(x) for x in flattened_pytree]
    return jnp.concat(flattened_pytree), TreeStructure(structure, leaf_shapes)

def batched_tree_ravel(pytree: dict):
    flattened_pytree, structure = jax.tree_flatten(pytree)
    flattened_pytree: List[chex.Array]
    leaf_shapes = [x.shape[1:] for x in flattened_pytree]
    flattened_pytree = [jnp.reshape(x, (x.shape[0], reduce(mul, x.shape[1:]))) for x in flattened_pytree]
    return jnp.hstack(flattened_pytree), TreeStructure(structure, leaf_shapes)


def tree_unravel(tree_structure: TreeStructure, vec: chex.Array):
    pytree_structure = tree_structure.pytree_structure
    leaf_shapes = tree_structure.leaf_shapes

    flattened_leaf_shapes = jnp.array([reduce(mul, s) for s in leaf_shapes])
    split_indexes = jnp.cumsum(flattened_leaf_shapes)
    leaves = jnp.split(vec, split_indexes)
    reshaped_leaves = []
    for (shape, leaf) in zip(leaf_shapes, leaves):
        reshaped_leaves.append(jnp.reshape(leaf, shape))
    return jax.tree_unflatten(pytree_structure, reshaped_leaves)


def plot_log_prob(ax: Axes, log_prob: Callable[[chex.Array], chex.Array], title: str):
    resolution = 100
    xline = jnp.linspace(-4, 4, resolution)
    yline = jnp.linspace(-4, 4, resolution)
    xgrid, ygrid = jnp.meshgrid(xline, yline)
    xyinput = jnp.hstack([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)])
    zgrid = log_prob(xyinput)
    #print(zgrid.min())
    #print(jnp.isnan(zgrid).any())
    zgrid = jnp.reshape(zgrid, (resolution, resolution))

    zgrid = jnp.exp(zgrid)
    
    
    ax.contourf(xgrid, ygrid, zgrid, levels=100)
    ax.set_title(title)


def wrap_log_prob_function(func: Callable[[chex.Array], chex.Array]) -> Callable[[chex.Array], chex.Array]:
    def f_wrapped(samples: chex.Array) -> chex.Array:
        log_prob = func(samples)
        f_wrapped._num_calls += log_prob.shape[0]
        return log_prob
    f_wrapped._num_calls = 0
    return f_wrapped

def get_log_prob_num_calls(func: Callable[[chex.Array], chex.Array]) -> int:
    return func._num_calls