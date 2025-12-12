import numpy as np


def random_starting_step(
        total_steps: int,
        interval: int,
        seed : int=None
    ):
    """Returns a random starting step number by uniformly sampling from
    {0, interval * 1, interval * 2, ..., total_steps - interval}
    Arguments:
    ---------
        total_steps : int
            Total number of steps.
        interval : int
            Interval that creates the sample space.
        seed : int
            Randomization seed
    """
    np.random.seed(seed)
    possible_starts = np.arange(0, total_steps, interval)
    sampled_step = np.random.choice(possible_starts)

    return sampled_step