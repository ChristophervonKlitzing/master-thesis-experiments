from bgflow.nn.flow.base import Flow
import zuko
from torch.distributions.utils import _sum_rightmost
import torch 

class ZukoFlow(Flow):
    def __init__(self, features: int, **kwargs):
        super().__init__()
        self._zuko_flow = zuko.flows.NSF(features=features, **kwargs)

    def _forward(self, z, temperature: float = 1.):
        flow = self._zuko_flow.forward()
        
        x, ladj = flow.transform.inv.call_and_ladj(z)
        ladj = _sum_rightmost(ladj, flow.reinterpreted)
        return x, torch.unsqueeze(ladj, -1)

    def _inverse(self, x, temperature: float = 1.):
        flow = self._zuko_flow.forward()
        z, ladj = flow.transform.call_and_ladj(x)
        ladj = _sum_rightmost(ladj, flow.reinterpreted)
        return z, -torch.unsqueeze(-ladj, -1)