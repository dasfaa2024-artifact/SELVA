from itertools import product

import numpy as np

from experiments.constants import scale
from experiments.profile import Profile
from experiments.utilities import gen_times, trunc_for, load_data, gen_samples
from samplers.betting_by_time import betting_factory


def get_times(size, cap_s, i):
    if cap_s == 'geo':
        b = 1.05 - min(i, 4) * 0.01 - max(i-4, 0) * 0.001
        times = gen_times(size // 3.3333, b, size)
    else:
        times = np.arange(size+1, dtype=np.int32)
    return times



def run_time_for(samples, repeat=1000):
    samples = gen_samples(samples, repeat)
    prof = Profile()
    delta = .002
    grid_num = 2000
    all_mtd = []
    for bet_s, cap_s in product(('vanilla', 'ada'), ('geo', 'seq')):
        mtd = f'{bet_s}_{cap_s}'
        print('running time exp.\n',mtd)
        cap, betting = betting_factory(mtd)
        gambler = cap(alpha=.05, grid_num=grid_num)
        prof.obj = gambler
        gambler.trunc_scale = trunc_for(cap_s, 0.01)
        all_size = []
        for si ,size in enumerate(scale):
            times = get_times(size, cap_s, si)
            ctn = []
            for i in range(repeat):
                with prof:
                    mu, t = betting(samples[i], times, .5, delta, grid_num,
                                    gambler)
                assert t == size
                ctn.append(prof.dt)
            all_size.append(ctn)
        all_mtd.append(all_size)
    return all_mtd

if __name__ == '__main__':
    label = load_data('/home/lg/VDBM/multi_predicate/selectivity_sampling/pseudo_label.pkl', 'm')
    state = run_time_for(label)
    np.save('run_time_fix_num_geo', np.array(state))


