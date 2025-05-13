from typing import Callable
from nflows import distributions, transforms, flows 
import bgflow as bg
import torch 

import matplotlib.pyplot as plt

from .utils.plot_log_prob import plot_probability
from .flows.train_normal import train_normal
from .flows.zuko_wrapper import ZukoFlow
from .targets.energy_potentials import U1

from bgflow import Energy

class CustomEnergy(Energy):
    def __init__(self, dim: int, log_prob_fn: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__(dim)
        self._log_prob_fn = log_prob_fn

    def _energy(self, x):
        return -self._log_prob_fn(x)



class RealNVP(bg.SequentialFlow):
    def __init__(self, dim, hidden):
        self.dim = dim
        self.hidden = hidden
        super().__init__(self._create_layers())
    
    def _create_layers(self):
        dim_channel1 =  self.dim//2
        dim_channel2 = self.dim - dim_channel1
        split_into_2 = bg.SplitFlow(dim_channel1, dim_channel2)
        
        layers = [
            # -- split
            split_into_2,
            # --transform
            self._coupling_block(dim_channel1, dim_channel2),
            bg.SwapFlow(),
            self._coupling_block(dim_channel2, dim_channel1),
            # -- merge
            bg.InverseFlow(split_into_2)
        ]
        return layers
    
    def _dense_net(self, dim1, dim2):
        return bg.DenseNet(
            [dim1, *self.hidden, dim2],
            activation=torch.nn.ReLU()
        )
    
    def _coupling_block(self, dim1, dim2):
        return bg.CouplingFlow(bg.AffineTransformer(
            shift_transformation=self._dense_net(dim1, dim2),
            scale_transformation=self._dense_net(dim1, dim2)
        ))
    

def run(cfg):
    print("Run experiment 'Adaptive smoothing'")

    dim = 2

    prior = bg.NormalDistribution(dim)
    target = CustomEnergy(2, U1)
    flow = ZukoFlow(dim) # RealNVP(dim, [128, 128, 128])

    # The BG is defined by a prior, target and a flow
    generator = bg.BoltzmannGenerator(prior, flow, target)

    z = generator.prior.sample(3)
    samples, log_det = generator.flow.forward(z)
    print(samples, log_det)

    fig, ax = plt.subplots()
    plot_probability(ax, lambda x: -generator.energy(x), title="test")
    plt.show()

    train_normal(
        generator=generator, 
        num_iterations=2000,
        eval_interval=100,
    )

    plt.close()
    fig, ax = plt.subplots()
    plot_probability(ax, lambda x: -generator.energy(x), title="test")
    plt.show()