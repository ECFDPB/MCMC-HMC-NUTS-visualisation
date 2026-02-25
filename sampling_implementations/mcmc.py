from .sampler_base import Sampler
from abc import ABC, abstractmethod


class TransitionKernel(ABC):
    @abstractmethod
    def configure(self, target_distribution, start_value, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, current_sample, *args, **kwargs):
        raise NotImplementedError()


class MCMC(Sampler):
    def __init__(self, transition_kernel: TransitionKernel):
        self.transition = transition_kernel

    def sample(self, target_distribution, num_samples, start_value, burn_in,
               transition_args=[], transition_kwargs={}):

        self.transition.configure(
            target_distribution,
            start_value,
            *transition_args,
            **transition_kwargs
        )

        # Burn in stage to get samples reflecting the target distribution.
        current_sample = start_value
        for i in range(burn_in):
            current_sample = self.transition(current_sample)

        # Sampling from the markov chain after the burn in stage.
        sample = []
        for i in range(num_samples):
            sample.append(self.transition(current_sample))
            current_sample = sample[i]

        return sample
