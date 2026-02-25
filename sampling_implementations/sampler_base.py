from abc import ABC, abstractmethod


class Sampler(ABC):
    @abstractmethod
    def sample(self, target, num_samples, *args, **kwargs):
        raise NotImplementedError()
