from typing import Callable, Union

import torch
from typing import TYPE_CHECKING

from ..model.mixture_model import MixtureModel
from ..model.diagonal_gaussian import DiagonalGaussian

from ..sample_db import SampleDB


if TYPE_CHECKING:
    from ...train import Trainer


def _rank(value: Union[float, torch.Tensor]):
    if isinstance(value, float):
        return 0
    else:
        return len(value.shape)
    
class VipsComponentAdaptation:
    """ This class implements the component adaptation procedure used by VIPS.

    See :cite:p:`Arenz2020`.

    Parameters:
        gmm_wrapper: :py:class:`GmmWrapper<gmmvi.models.gmm_wrapper.GmmWrapper>`
            The wrapped model where we want to adapt the number of components.

        sample_db: :py:class:`SampleDB<gmmvi.optimization.sample_db.SampleDB>`
            The sample database can be used to select candidate locations for adding a new component, without having
            to perform additional queries to the target distribution.

        target_distribution: :py:class:`LNPDF<gmmvi.experiments.target_distributions.lnpdf.LNPDF>`
             The target distribution can be used to evaluate candidate locations for adding a new component.

        prior_mean: tf.Tensor
            A one dimensional tensor of size num_dimensions, specifying the mean of the Gaussian that we can use to sample
            candidate locations for adding a new component.

        initial_cov: tf.Tensor
            A two-dimensional tensor of size num_dimensions x num_dimensions, specifying the covariance of the Gaussian
            that we can use to sample candidate locations for adding a new component.

        del_iters: int
            minimum number of updates a component needs to have received, before it is considered as candidate for deletion.

        add_iters: int
            a new component will be added every *add_iters* iterations

        max_components: int
            do not add components, if the model has at least *max_components* components

        num_database_samples: int
            number of samples from the :py:class:`SampleDB<gmmvi.optimization.sample_db.SampleDB>` that are used for
            selecting a good initial mean when adding a new component.

        num_prior_samples: int
            number of samples from the prior distribution that are used for selecting a good initial mean when adding a
            new component.
    """
    def __init__(self, trainer: 'Trainer', model: MixtureModel, sample_db: SampleDB, target_lnpdf: Callable[[torch.Tensor], torch.Tensor], prior_mean: Union[float, torch.Tensor],
                 initial_cov: Union[float, torch.Tensor], del_iters: int, add_iters: int, max_components: int,
                 thresholds_for_add_heuristic: float, min_weight_for_del_heuristic: float,
                 num_database_samples: int, num_prior_samples: int):
        self.trainer = trainer
        self.model = model
        if (prior_mean is not None) and (initial_cov is not None):
            if _rank(initial_cov) == 0:
                initial_cov = initial_cov * torch.ones(model.num_dimensions)
            if _rank(prior_mean) == 0:
                prior_mean = prior_mean * torch.ones(model.num_dimensions)
            
            self.prior = DiagonalGaussian(prior_mean, initial_cov)
        else:
            self.prior = None

        self.num_prior_samples = num_prior_samples
        self.target_lnpdf = target_lnpdf
        self.sample_db = sample_db
        self.del_iters = del_iters
        self.add_iters = add_iters
        self.max_components = max_components
        self.num_db_samples = num_database_samples
        self.num_calls_to_add_heuristic = 0
        self.thresholds_for_addHeuristic = torch.tensor(thresholds_for_add_heuristic, dtype=torch.float32)
        self.min_weight_for_del_heuristic = min_weight_for_del_heuristic

        self.reward_improvements = 0.0
        self.filter_delay = int(torch.math.floor(self.del_iters / 3))
        gaussian = torch.distributions.Normal(0.0, self.del_iters / 8)
        log_kernel: torch.Tensor = gaussian.log_prob(torch.range(start=-self.filter_delay, end=self.filter_delay, dtype=torch.float32))
        normalized_kernel: torch.Tensor = (log_kernel - torch.logsumexp(log_kernel, 0)).exp()
        self.kernel = torch.reshape(normalized_kernel, [-1, 1, 1])

    def adapt_number_of_components(self, iteration: int):
        """ This method may change the number of components, either by deleting bad components that have low weights,
         or by adding new components.

         Parameters:
             iteration: int
                The current iteration, used to decide whether a new component should be added.
         """

        if iteration > self.del_iters:
            self.delete_bad_components()
        if iteration > 1 and iteration % self.add_iters == 0:
            if self.model.get_num_components() < self.max_components:
                self.add_new_component()

    def add_at_best_location(self, samples: torch.Tensor, target_lnpdfs: torch.Tensor):
        """ Find the most promising :cite:p:`Arenz2020` location among the provided samples for adding a new component,
        that is, a new component will be added with mean given by one of the provided samples.

        Parameters:
            samples: torch.Tensor
                candidate locations for initializing the mean of the new component

            target_lnpdfs: torch.Tensor
                for each candidate location, this tensor contains the log-density under the (unnormalized) target
                distribution.
        """
        iter = self.num_calls_to_add_heuristic % self.thresholds_for_addHeuristic.shape[0]
        model_log_densities = self.model.log_density(samples)
        init_weight = 1e-29
        a = torch.rand([1])
        if self.prior is not None:
            des_entropy = self.model.get_average_entropy() * a + self.prior.entropy() * (1 - a)
        else:
            des_entropy = self.model.get_average_entropy()
        max_logdensity = torch.max(model_log_densities)
        rewards = target_lnpdfs - torch.max(torch.concat([
            torch.unsqueeze(max_logdensity - self.thresholds_for_addHeuristic[iter], 0),
            model_log_densities,
        ]))
        new_mean = samples[torch.argmax(rewards)]
        H_unscaled = 0.5 * self.model.num_dimensions * (torch.math.log(2.0 * torch.pi) + 1)
        c = torch.math.exp((2 * (des_entropy - H_unscaled)) / self.model.num_dimensions)
        new_cov = c * torch.ones(self.model.num_dimensions)

        self.trainer.add_component(init_weight, new_mean, new_cov, torch.reshape(self.thresholds_for_addHeuristic[iter], [1]),
                                 torch.reshape(des_entropy, [1]))

    def select_samples_for_adding_heuristic(self):
        """ Select a set of samples used as candidates for initializing the mean of the new component.

        Returns:
            tuple(tf.Tensor, tf.Tensor, tf.Tensor):

            **samples** - the selected candidate locations

            **target_lnpdfs** - log-densities of the *samples* under the unnormalized target distribution

            **prior_samples** - additional samples drawn from a prior, which have not yet been evaluated on the
            target distribution.
        """
        self.num_calls_to_add_heuristic += 1
        samples, target_lnpdfs = self.sample_db.get_random_sample(self.num_db_samples)
        prior_samples = torch.zeros((0, self.model.num_dimensions), dtype=torch.float32)

        if self.num_prior_samples > 0:
            prior_samples = self.prior.sample(self.num_prior_samples)
            self.sample_db.num_samples_written += self.num_prior_samples
        return samples, target_lnpdfs, prior_samples

    def add_new_component(self):
        """ This method adds a new component by first selecting a set of candidate locations and the choosing the most
        promising one using the procedure of VIPS :cite:p:`Arenz2020`.
        """
        samples, target_lnpdfs, prior_samples = self.select_samples_for_adding_heuristic()
        if self.num_prior_samples > 0:
            samples = torch.concat((samples, prior_samples), 0)
            target_lnpdfs = torch.concat((target_lnpdfs, self.target_lnpdf(prior_samples)), 0)
        self.add_at_best_location(samples, target_lnpdfs)

    def delete_bad_components(self):
        """ Components are deleted, if all the following criteria are met received:

        1. It must have received at least *del_iters* updates

        2. It must not have improved significantly during the last iterations. In contrast to VIPS, we use a Gaussian
           filter to smooth the rewards of the component, to be more robust with respect to noisy target distributions.

        3. It must have very low weight, such that the effects on the model are negligible.
        """

        # estimate the relative improvement for every component with respect to
        # the improvement it would need to catch up (assuming linear improvement) with the best component
        kernel_size = self.kernel.shape.numel()
        current_smoothed_reward = torch.mean(
            self.trainer.reward_history[:, -kernel_size:] * torch.reshape(self.kernel, [1, -1]), dim=1)
        
        print(self.trainer.reward_history.shape)
        print(self.trainer.reward_history[:, -kernel_size - self.del_iters : -self.del_iters].shape)
        print(torch.reshape(self.kernel, [1, -1]).shape)
        
        old_smoothed_reward = torch.mean(
            self.trainer.reward_history[:, -kernel_size - self.del_iters : -self.del_iters]
            * torch.reshape(self.kernel, [1, -1]), dim=1)

        old_smoothed_reward -= torch.max(current_smoothed_reward)
        current_smoothed_reward -= torch.max(current_smoothed_reward)
        reward_improvements = (current_smoothed_reward - old_smoothed_reward) / torch.abs(old_smoothed_reward)
        self.reward_improvements = reward_improvements
        # compute for each component the maximum weight it had within the last del_iters,
        # or that it would have gotten when we used greedy updates
        max_actual_weights = torch.max(self.trainer.weight_history[:, -kernel_size - self.del_iters : -1], dim=1)[0]
        max_greedy_weights = torch.max(torch.exp(
                self.trainer.reward_history[:, -kernel_size - self.del_iters:] 
                - torch.logsumexp(self.trainer.reward_history[:, -kernel_size - self.del_iters:], dim=0, keepdim=True)
        ), dim=1)[0]

        max_weights = torch.maximum(max_actual_weights, max_greedy_weights)

        is_stagnating = reward_improvements <= 0.4
        is_low_weight = max_weights < self.min_weight_for_del_heuristic
        is_old_enough = self.trainer.reward_history[:, -self.del_iters] != -torch.finfo(torch.float32).max 
        is_bad = torch.all(torch.stack([is_stagnating, is_low_weight, is_old_enough]), dim=0)
        # is_bad = tf.reduce_all((is_stagnating, is_low_weight, is_old_enough), dim=0)
        bad_component_indices = torch.squeeze(torch.nonzero(is_bad), dim=1)

        if bad_component_indices.shape.numel() > 0:
            for idx in torch.sort(bad_component_indices, descending=True):
                self.trainer.remove_component(idx)


