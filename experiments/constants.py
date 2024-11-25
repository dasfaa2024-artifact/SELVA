from itertools import product

import numpy as np

epsilons = np.logspace(*np.log10([.01, .25]), 13)

eps_dic = {d: i for i, d in enumerate(epsilons)}

starts = np.logspace(*np.log10([7000, 8.1]), 13, dtype=int) // 2
starts[-1] = 6
# for cv
# starts = np.logspace(*np.log10([7000, 8.1]), 13, dtype=int)

betas = np.logspace(*np.log10([1.01, 1.09]), 13)
# betas = np.logspace(*np.log10([1.023, 1.109]), 13)

deltas = np.linspace(.01, .21, 11)
truncs = [
    [.05, .05, .05, .075, .1, .9, .9, .9, .9, .9, .9, .9, .75],
    [.025, .025, .05, .05, .075, .075, .1, .25, .25, .25, .5, .5, .75]]
trunc_dic = {f'{cap}': truncs[i] for i,  cap in
             enumerate(('geo', 'seq'))}

eps_m = {e: (s, b) for e, s, b in zip(epsilons, starts, betas)}
scale = [100, 500, 1000, 5000, 10000, 50000, 100000]

sels = [0., .0513, .1063, .1572, .2034, .2483, .31 , .3577, .4008,.4512, .5087,
        .55 , .5904 , .6513, .7141, .7486, .7984, .8504, .8996, .9422, 1.]

sample_nums = np.logspace(*np.log10([50, 50000]), 11)

methods = list(product(('vanilla', 'ada'), ('geo', 'seq')))
