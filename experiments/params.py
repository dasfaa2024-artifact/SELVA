import argparse
import pickle
from pathlib import Path
from functools import partial

import numpy as np

from experiments.constants import epsilons, methods, deltas, sels, eps_m
from experiments.profile import Profile
from experiments.simulation import repeat_cv_sampling, repeat_multicv_sampling
from experiments.utilities import (load_data, gen_times_by_epsilon,
                                   trunc_for, gen_samples, gen_times)
from samplers.betting_by_time import betting_factory


def best_trunc_size(samples, times, truncs, gambler,
                    betting, repeat=100, **kwargs):
    ctn = []
    for trunc in truncs:
        gambler.trunc_scale = trunc
        count = 0
        for _ in range(repeat):
            X = np.random.permutation(samples)
            _, t = betting(X, times, gambler=gambler, **kwargs)
            count += t
            gambler.reset()
        ctn.append(count)
    return truncs[np.argmin(ctn)]

def multi_trunc_size(samples, betting, gambler, pof, cap_type):
    trus = [0.9, 0.8, 0.75, 0.5, 0.25, 0.1, 0.075, 0.05, 0.025, 0.01]
    best = []
    for e in epsilons:
        times = gen_times_by_epsilon(e, cap_type)
        best_tru = best_trunc_size(samples, times, trus, gambler, betting,
                                   prior_mean=.5, epsilon=e, grid_num=1000)
        best.append(best_tru)
        print(e, end=' ')
    return best

def sample_num(samples, betting, gambler, pof, cap_type, repeat=1000):
    samples = [np.random.permutation(samples) for _ in range(repeat)]
    all_eps = []
    for eps in epsilons:
        trunc = trunc_for(cap_type, eps)
        gambler.trunc_scale = trunc
        times = gen_times_by_epsilon(eps, cap_type)
        t_total = 0
        for i in range(repeat):
            _, t = betting(samples[i], times, gambler=gambler,
                           prior_mean=.5, epsilon=eps, grid_num=1000)
            gambler.reset()
            t_total += t
        all_eps.append(t_total / repeat)
    return all_eps

def one_run(betting, gambler, samples, times, e, repeat, pof, cv=False):
    ctn = []
    for i in range(repeat):
        X = samples[i][:times[-1] + 1].astype(np.float32, copy=False)
        if cv:
            candi = np.unique(X)
            gambler.set_cand(candi)
        with pof:
            mu, t = betting(X, times, .5, e, repeat, gambler)
        # gambler.reset()
        ctn.append((mu, t, pof.dt))
    return ctn


def multi_eps(betting, gambler, pof, cap_s, samples, eps, repeat=1000, cv=False):
    all_eps = []
    for e in eps:
        times = gen_times_by_epsilon(e, cap_s, 30000)
        trunc = trunc_for(cap_s, e)
        gambler.trunc_scale = trunc
        run_res = one_run(betting, gambler, samples, times, e, repeat, pof, cv)
        all_eps.append(run_res)
        print(e, end=' ')
    return all_eps

def multi_delta(betting, gambler, pof, cap_s, samples, delts, eps=.01,
                repeat=1000, cv=False):
    gambler.trunc_scale = trunc_for(cap_s, eps)
    all_alpha = []
    times = gen_times_by_epsilon(eps, cap_s, 40000)
    for delta in delts:
        times[1:] -= round(25**((delta-.01) * 11))
        gambler.set_delta(delta)
        run_res = one_run(betting, gambler, samples, times, eps, repeat, pof,
                          cv)
        all_alpha.append(run_res)
        print(delta, end=' ')
    return all_alpha

def gen_time_for_mutil_sel(sel, eps, cap):
    if cap == 'seq':
        return np.arange(100000, dtype=np.int32)
    s, b = eps_m[eps]
    mean = 0.714
    if sel < .5:
        sel = 1 - sel
    s *= (1 + mean - sel) ** 2
    b = b**(0.4 + sel)
    return gen_times(s, b, 25000)

def multi_selectivity(betting, gambler, pof, cap_s, samples_aug, sel,
                      eps=.01, repeat=1000, cv=False):
    gambler.trunc_scale = trunc_for(cap_s, eps)
    all_sam = []
    mean = [.1010, .2096, .2875, .4051, .5080, .6162, .7041, .7835, .9072]
    count = 0
    for cur_samples, s in zip(samples_aug, sel):
        pof.pmean = mean[count]
        times = gen_time_for_mutil_sel(s, eps, cap_s)
        run_res = one_run(betting, gambler, cur_samples, times, eps, repeat,
                          pof, cv=cv)
        all_sam.append(run_res)
        print(s, end=' ')
        count += 1
    return all_sam

def multi_mth_run(func, mths, prior_mean=.5, prior_var=.25,
                  num=1, cand_num=2, save_name=None):
    pof = Profile(0., prior_mean, prior_var, num)
    all_mtd = []
    for bet_s, cap_s in mths:
        mtd = f'{bet_s}_{cap_s}'
        print(mtd)
        cap_process, betting = betting_factory(mtd)
        gambler = cap_process(delta=.05, prior_mean=prior_mean,
                              prior_var=prior_var,
                              num=num, cand_num=cand_num)
        pof.obj = gambler
        all_mtd.append(func(betting, gambler, pof, cap_s))
        print()
    all_mtd = np.array(all_mtd)
    if save_name:
        np.save(save_name, all_mtd)
    return all_mtd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', '-r', type=str,
                        required=False, default='vdelt',
                        choices=['veps', 'vdelt', 'vsel', 'probtrunc',
                                 'probsn', 'cv', 'cmp'])
    parser.add_argument('--inpath', '-i', type=Path, required=True,
                        help="The path of file containing value of evaluated predicates.")
    parser.add_argument('--repeat', '-p', type=int, default=1000, required=False,
                        help='rounds of experimets.')
    args = parser.parse_args()
    label = load_data(args.inpath, 'm')
    repeat = args.repeat
    match args.run:
        case 'probtrunc':
            multi_mth_run(partial(multi_trunc_size, label), methods,
                          save_name='prob_trunc')
        case 'probsn':
            multi_mth_run(partial(sample_num, label), methods,
                          save_name='prob_sample_num')
        case 'veps':
            func = partial(multi_eps, samples=gen_samples(label, repeat, 30000), eps=epsilons, repeat=repeat)
            stat = multi_mth_run(func, methods[2:3], save_name='multi_epsilon')
        case 'vdelt':
            func = partial(multi_delta, samples=gen_samples(label, repeat, 40000),
                           delts=deltas, repeat=repeat)
            stat = multi_mth_run(func, methods, save_name='delta')
        case 'vsel':
            with open(args.inpath, 'rb') as f:
                vsamples = pickle.load(f)
            samples_aug = [gen_samples(sam, repeat) for sam in vsamples[0]]
            func = partial(multi_selectivity, samples_aug=samples_aug, sel=sels)
            multi_mth_run(func, methods, save_name='multi_sel.npy')

        case 'cv':
            labels = load_data(args.inpath, 'mx')
            samples = repeat_cv_sampling(labels[0].astype(np.float32),
                                         labels[1].astype(np.float32), 40000,
                                         repeat)
            func = partial(multi_delta, samples=samples, delts=deltas,
                           repeat=repeat, cv=True)
            multi_mth_run(func, methods, 0.714,
                          0.2, 100, 4, save_name='cv_delta')
            func = partial(multi_eps, samples=samples, eps=epsilons,
                           repeat=repeat, cv=True)
            stat = multi_mth_run(func, methods, 0.714, 0.2, 100,
                                 4, save_name='cv_eps')
            with open(args.inpath, 'rb') as f:
                pcv = pickle.load(f)
            samples_aug = [repeat_cv_sampling(p.astype(np.float32),
                                              cv.astype(np.float32),
                                              30000, repeat)
                           for (p, cv) in pcv]
            func = partial(multi_selectivity, samples_aug=samples_aug,
                           sel=np.arange(1,10) *0.1, repeat=repeat, cv=True)
            multi_mth_run(func, methods, 0.714, 0.2, 100, 4,
                          save_name='cv_sel')
        case 'cmp':
            with open(args.inpath, 'rb') as f:
                labels = pickle.load(f)
            model_hierarchical = (('dfe_tasti', 'dfe', 'ferc', 'ferm'),
                                  ('fd_tasti', 'fd', 'dffd'),
                                  ('n_tasti', 'n', 's', 'm', 'l', 'x'))
            time_ctn = []
            cap_process, betting = betting_factory('ada_geo')
            for task in model_hierarchical:
                task_ctn = []
                prof = Profile()
                gambler = cap_process(delta=.05, cand_num=2)
                prof.obj = gambler
                gambler.trunc_scale = 0.9
                model = task[0]
                samples = gen_samples(labels[model], repeat, 1000)
                times = gen_times(400, 1.03, 1000)
                task_ctn.append(one_run(betting, gambler, samples, times, 0.0467, repeat, prof))
                prof.pfnum = 100
                for i in range(1, len(task)):
                    m = labels[task[i]]
                    cv = labels[task[i-1]]
                    prof.pmean = np.mean(np.array(task_ctn[-1])[:, 0])
                    samples = repeat_cv_sampling(m.astype(np.float32),
                                                 cv.astype(np.float32),1000,
                                                 repeat)
                    times = gen_times(90, 1.04, 1000)
                    task_ctn.append(one_run(betting, gambler, samples, times, 0.0467, repeat, prof,True))
                time_ctn.extend(task_ctn)
            np.save('cmp_emo', np.array(time_ctn))



