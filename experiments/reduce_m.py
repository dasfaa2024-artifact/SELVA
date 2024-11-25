import numpy as np
from numba import njit, float32, int32

from experiments.utilities import load_data, trunc_for, gen_times_by_epsilon
from samplers.betting_by_time import betting_factory
from samplers.betting_by_time.betting_strategies import intersect


@njit((float32[:], float32[:], int32, float32, float32))
def gen_bound(samples, capitals, gird_num, trunc_scale, threshold):
    bound = np.empty((len(samples)+1, 2))
    bound[0] = 0., 1.
    cum_capitals = np.ones((gird_num+1, 2))
    cs_bound = np.array([0., 1.], dtype=np.float32)
    for i, (x, lbd) in enumerate(zip(samples, capitals), 1):
        for mi in range(gird_num+1):
            m = mi / gird_num
            if abs(x - m) < 1e-9:
                # use convention that inf * 0 = 0. we still have
                # a martingale under the null. thus, 1 + 0 = 1, cum_capital unchanged
                continue
            cum_capital = cum_capitals[mi, 0]
            if cum_capital > 1e-16:
                lbd_m = min(max(lbd, -trunc_scale / (1 + 1e-9 - m)), trunc_scale / (m + 1e-9))
                cum_capital *= 1. + lbd_m * (x - m)
                cum_capitals[mi, 0] = cum_capital
            cum_capital = cum_capitals[mi, 1]
            if cum_capital > 1e-16:
                lbd_m = min(max(-lbd, -trunc_scale / (1 + 1e-9 - m)), trunc_scale / (m + 1e-9))
                cum_capital *= 1. + lbd_m * (x - m)
                cum_capitals[mi, 1] = cum_capital
        rang_idx = np.flatnonzero(np.sum(cum_capitals, axis=1) < threshold)
        intersect(cs_bound, rang_idx[0] / gird_num, rang_idx[-1] / gird_num)
        bound[i] = cs_bound
    return bound



def probe_pos(samples):
    # seed = generate_big_prime()
    # print('seed:', seed)
    rng = np.random.default_rng(2802154141)
    mth = 'ada_geo'
    cap_process, betting = betting_factory(mth)
    eps = .0225
    trunc = trunc_for('geo', eps)
    gambler = cap_process(delta=.05, trunc_scale=trunc)
    times = gen_times_by_epsilon(eps, 'geo')
    # times = gen_times(500, 1.011)
    samples = rng.permutation(samples)
    X1 = samples[:times[-1]+1]
    mu, t, pos_s1 = betting(X1, times, 0.5, eps, 1000, gambler)
    print('times:', t)
    pos1, pos2 = post_process(pos_s1, np.copy(gambler.cum_cap_pos), times)
    idx = np.searchsorted(times, t)
    gambler.add_sample(samples[times[idx]:times[idx+1]])
    capitals = gambler.capitals[:gambler.s_ptr]
    capitals = np.repeat(capitals, np.diff(times[:idx+2]))
    bound = gen_bound(X1[:times[idx+1]], capitals, gambler.grid_num, trunc, gambler.threshold)
    return (pos1, pos2), bound * gambler.grid_num

def control_pos(samples):
    rng = np.random.default_rng(2802154141)
    mth = 'vanilla_seq'
    cap_process, betting = betting_factory(mth)
    eps = .0225
    trunc = trunc_for('seq', eps)
    gambler = cap_process(delta=.05, trunc_scale=trunc)
    samples = rng.permutation(samples)
    mu, t, bounds = betting(samples, np.arange(3103, dtype=np.int32), 0.5, eps, 1000, gambler)
    print('times:', t)
    bounds[0] -= 1.
    bounds[-1] += 1.
    np.save('control_m', np.clip(bounds, 0, 1))

def post_process(pos1, pos2, times):
    pos1 = np.max(pos1, axis=1)
    pos2 = np.max(pos2, axis=1)
    for i in range(pos1.shape[0]):
        pos1[i] = times[pos1[i]]
        pos2[i] = times[pos2[i]]
    return pos1, pos2


if __name__ == '__main__':
    label = load_data('/home/lg/VDBM/multi_predicate/selectivity_sampling/pseudo_label.pkl', 'm')
    pos, bo = probe_pos(label)
    bo = bo.T
    bo[0] -= 1
    bo[1] += 1
    bo = np.clip(bo, 0, 1000)
    np.savez('cs', pos=pos, bo=bo)

    control_pos(label)