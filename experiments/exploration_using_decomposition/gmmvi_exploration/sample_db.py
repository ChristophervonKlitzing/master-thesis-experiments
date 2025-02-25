from math import pi
import torch 

class SampleDB:
    """ A database for storing samples and meta-information.

    Along the samples, we also store

    1. The parameters of the Gaussian distribution that were used for obtaining each sample

    2. log-density evaluations of the target distribution, :math:`\\log p(\\mathbf{x})`

    3. (if available), gradients of the log-densites of the target distribution,
       :math:`\\nabla_\\mathbf{x} \\log p(\\mathbf{x})`

    Parameters:
        dim: int
            dimensionality of the samples to be stored

        diagonal_covariances: bool
            True, if the samples are always drawn from Gaussians with diagonal covariances (saves memory)

        keep_samples: bool
            If this is False, the samples are not actually stored

        max_samples: int
            Maximal number of samples that are stored. If adding new samples would exceed this limit, every N-th sample
            in the database gets deleted.
    """
    def __init__(self, dim, keep_samples, max_samples=None):
        self._dim = dim
        self.keep_samples = keep_samples

        self.samples = torch.zeros((0, dim))
        self.target_lnpdfs = torch.zeros(0)
        self.num_samples_written = 0
        self.max_samples = max_samples

    @staticmethod
    def build_from_config(config, num_dimensions):
        """ A static method to conveniently create a :py:class:`SampleDB` from a given config dictionary.

        Parametes:
            config: dict
                The dictionary is typically read from YAML a file, and holds all hyperparameters.

            num_dimensions: int
                dimensionality of the samples to be stored
        """
        return SampleDB(num_dimensions, config["model_initialization"]["use_diagonal_covs"],
                             config["use_sample_database"],
                             config["max_database_size"])

    def remove_every_nth_sample(self, N: int):
        """ Deletes Every N-th sample from the database and the associated meta information.

        Parameters:
            N: int
                abovementioned N
        """
        self.samples = self.samples[::N]
        self.target_lnpdfs = self.target_lnpdfs[::N]

    def add_samples(self, samples: torch.Tensor, target_lnpdfs: torch.Tensor):
        """ Add the given samples to the database.

        Parameters:
            samples: tf.Tensor
                a two-dimensional tensor of shape num_samples x num_dimensions containing the samples to be added.
            target_lnpdfs: tf.Tensor
                a one-dimensional tensor containing the log-densities of the (unnormalized) target distribution,
                :math:`\\log p(\\mathbf{x})`.

            target_grads: tf.Tensor
                a two-dimensional tensor containing the gradients of the log-densities of the (unnormalized) target
                distribution, :math:`\\nabla_{\\mathbf{x}} \\log p(\\mathbf{x})`.
        """
        if self.max_samples is not None and samples.shape[0] + samples.shape[0] > self.max_samples:
            self.remove_every_nth_sample(2)
        self.num_samples_written += samples.shape[0]
        if self.keep_samples:
            self.samples = torch.concat((self.samples, samples), dim=0)
            self.target_lnpdfs = torch.concat((self.target_lnpdfs, target_lnpdfs), dim=0)
        else:
            self.samples = samples
            self.target_lnpdfs = target_lnpdfs

    def get_random_sample(self, N: int):
        """ Get N random samples from the database.

        Parameters:
            N: int
                abovementioned N

        Returns:
            tuple(tf.Tensor, tf.Tensor)

            **samples** - the chosen samples

            **target_lnpdfs** - the corresponding log densities of the target distribution
        """
        chosen_indices = torch.randperm(self.samples.shape[0])[:N]
        return self.samples[chosen_indices], self.target_lnpdfs[chosen_indices]
