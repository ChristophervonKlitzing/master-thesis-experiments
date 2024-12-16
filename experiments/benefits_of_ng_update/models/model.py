from typing import Callable, NamedTuple, Tuple, Type
import haiku as hk
from abc import ABC, abstractmethod
import chex


class Model(hk.Module, ABC):
    def __init__(self, name = None):
        super().__init__(name)

    @abstractmethod
    def log_prob(self, x: chex.Array):
        raise NotImplementedError
    
    @abstractmethod
    def sample(self, key: chex.PRNGKey, num_samples: int) -> chex.Array:
        raise NotImplementedError
    
    def sample_and_log_prob(self, key: chex.PRNGKey, num_samples: int) -> Tuple[chex.Array, chex.Array]:
        x = self.sample(key, num_samples)
        log_prob = self.log_prob(x)
        return x, log_prob


class ModelOp(NamedTuple):
    init: Callable[[chex.PRNGKey, chex.Array], hk.MutableParams]
    log_prob: Callable[[hk.MutableParams, chex.Array], chex.Array]
    sample: Callable[[hk.MutableParams, chex.PRNGKey, int], chex.Array]
    sample_and_log_prob: Callable[[hk.MutableParams, chex.PRNGKey, int], Tuple[chex.Array, chex.Array]]


def create_model(model_cls: Type[Model], dim: int):
    def log_prob(x):
        m: Model = model_cls(dim=dim)
        return m.log_prob(x)
    
    def sample(key: chex.PRNGKey, num_samples: int) -> chex.Array:
        m: Model = model_cls(dim=dim)
        return m.sample(key, num_samples)
    
    def sample_and_log_prob(key: chex.PRNGKey, num_samples: int) -> Tuple[chex.Array, chex.Array]:
        m: Model = model_cls(dim=dim)
        return m.sample_and_log_prob(key, num_samples)
    
    log_prob_transform = hk.without_apply_rng(hk.transform(log_prob))
    sample_transform = hk.without_apply_rng(hk.transform(sample))
    sample_and_log_prob_transform = hk.without_apply_rng(hk.transform(sample_and_log_prob))

    def init(key: chex.PRNGKey, sample: chex.Array):
        return log_prob_transform.init(key, sample)
    
    return ModelOp(
        init=init,
        log_prob=log_prob_transform.apply,
        sample=sample_transform.apply,
        sample_and_log_prob=sample_and_log_prob_transform.apply,
    )
    