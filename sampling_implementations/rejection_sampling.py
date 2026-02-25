from .sampler_base import Sampler
import numpy as np


class RejectionSampler(Sampler):

    def __init__(self):
        raise NotImplementedError()

    def sample(self, target, num_iterations, approximiate_sampler,
               approximate_pdf, bounding_constant, random_state=None,
               *args, **kwargs):

        rng = np.random.RandomState(random_state)
        samples = []
        for i in range(num_iterations):
            current_value = approximiate_sampler()
            if target(current_value) / \
                    (bounding_constant*approximate_pdf(current_value)) > \
                    rng.uniform():
                samples.append(current_value)

        return samples
