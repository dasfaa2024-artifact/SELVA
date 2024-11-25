from experiments.constants import epsilons
from experiments.utilities import gen_times, gen_times_by_epsilon
from samplers.betting_by_time import betting_factory

if __name__ == '__main__':
    import numpy as np
    bet_str = 'ada'
    cap_str = 'geo'
    if cap_str == 'seq':
        times = np.arange(40000, dtype=np.int32)
    else:
        times = gen_times(35, 1.04)
    mean = 0.72353
    repeat = 50
    cap_process, betting = betting_factory(mtd=f'{bet_str}_{cap_str}')
    gambler = cap_process(alpha=.5, trunc_scale=.1)

    for e, in epsilons:
        total = 0
        times = gen_times_by_epsilon(e, cap_str)
        for _ in range(repeat):
            x = np.random.binomial(1, mean, times[-1]+1).astype(np.float32)
            mu, t = betting(x, times, prior_mean=5, delta=e,
                            grid_num=1000, gambler=gambler)
            total += t
            gambler.reset()
        print('delta:', e, 'avg_num:', total / repeat)