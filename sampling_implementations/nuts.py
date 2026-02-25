from .mcmc import TransitionKernel


class NUTS(TransitionKernel):
    def __init__(self):
        raise NotImplementedError()

    def __call__(self, current_sample, *args, **kwargs):
        raise NotImplementedError()

    def configure(self, target_distribution, start_value, *args, **kwargs):
        raise NotImplementedError()
