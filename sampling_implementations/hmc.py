import numpy as np
from .mcmc import TransitionKernel


class HMC(TransitionKernel):
    def __init__(self, random_seed=None):
        self.rng = np.random.RandomState(random_seed)

    def configure(self, target_distribution, start_value, log_lik_grad,
                  step_size, num_steps, *args, **kwargs):
        self.target_distribution = target_distribution
        self.current_momentum = self.rng.multivariate_normal(
            mean=np.zeros_like(start_value),
            cov=np.identity(len(start_value))
        )
        self.log_dist_grad = log_lik_grad
        self.step_size = step_size
        self.num_steps = num_steps

    def __call__(self, current_sample):
        proposed_sample, proposed_momentum = current_sample, \
            self.current_momentum

        for i in range(self.num_steps):
            proposed_sample, proposed_momentum = self.leapfrog(
                proposed_sample, proposed_momentum)

        normal = lambda m: np.exp(-0.5*np.linalg.norm(m)**2) /\
            (2*np.pi)**(len(proposed_momentum)/2)

        metropolis_ratio = np.min([
            1,
            self.target_distribution(proposed_sample) *
            normal(proposed_momentum) /
            self.target_distribution(current_sample) *
            normal(self.current_momentum)
        ])

        if self.rng.uniform() <= metropolis_ratio:
            current_sample, self.current_momentum = proposed_sample, \
                -proposed_momentum

        return current_sample

    def leapfrog(self, proposed_sample, proposed_momentum):
        proposed_momentum = proposed_momentum +\
            self.log_dist_grad(proposed_sample) * (self.step_size/2)

        proposed_sample = proposed_sample + self.step_size*proposed_momentum

        proposed_momentum += self.log_dist_grad(proposed_sample) * \
            (self.step_size/2)

        return proposed_sample, proposed_momentum
