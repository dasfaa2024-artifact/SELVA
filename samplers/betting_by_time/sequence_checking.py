import math

from numba.experimental import jitclass
from numba import int32, float32, njit, float64

import numpy as np

from samplers.betting_by_time.utilities import cal_c


@jitclass
class Lambda:
    cum: float32
    cum_diff2: float32
    t: int32
    alpha: float32
    c: float32

    def __init__(self, prior_mean=0.5, prior_var=0.25, num=1, alpha=0.05):
        self.cum = prior_mean * num
        self.cum_diff2 = prior_var * num
        self.t = num
        self.alpha = alpha
        self.c = cal_c(alpha)

    def reset(self, pior_mean, pior_diff2, num):
        self.t = num
        self.cum = pior_mean * num
        self.cum_diff2 = pior_diff2 * num

    def set_delta(self, delta):
        self.c = cal_c(delta)

    def advance(self, x: float32):
        t = self.t
        sigma2 = self.cum_diff2 / t
        lbd = math.sqrt(self.c / (sigma2 * t * math.log(t+1)))
        self.cum_diff2 += (x - self.cum / t) ** 2
        self.cum += x
        self.t = t + 1
        return lbd


@njit((float32, float32, float32, float64, float32), fastmath=True)
def seq_single_bet_on(trunc_scale, m, sample, cum_cap, capital):
    if abs(sample - m) < 1e-9 or cum_cap < 1e-16:
        # use convention that inf * 0 = 0. we still have
        # a martingale under the null. thus, 1 + 0 = 1, cum_capital unchanged
        return cum_cap
    lbd_m = min(max(capital, -trunc_scale / (1 + 1e-9 - m)), trunc_scale / (m + 1e-9))

    cum_cap *= 1. + lbd_m * (sample - m)
    return cum_cap

@jitclass
class SequenceCheckingCapital:
    cap_mine: Lambda
    cum_cap_twins: float64[:, :]
    cum_cap_pos: int32[:, :]
    capitals: float32[:]
    samples: float32[:]
    trunc_scale: float32
    threshold: float32
    grid_num: int32
    s_ptr: int32

    def __init__(self, delta=.05, trunc_scale=.5, grid_num=1000,
                 prior_mean=0.5, prior_var=0.25, num=1, cand_num=2):
        # cum[0] means we bet on mean > m, cum[1] means we bet on mean < m
        self.cum_cap_twins = np.ones((grid_num + 1, 2))
        self.cum_cap_pos = np.zeros((grid_num + 1, 2), dtype=int32)
        self.cap_mine = Lambda(prior_mean, prior_var, num, delta * 0.5)
        self.trunc_scale = trunc_scale
        self.threshold = 1 / delta
        self.samples = np.zeros(100100, dtype=float32)
        self.capitals = np.zeros(100100, dtype=float32)
        self.s_ptr = 0
        self.grid_num = grid_num

    def reset(self, prior_mean=0.5, prior_diff2=0.25, num=1):
        self.samples[:self.s_ptr] = 0.
        self.s_ptr = 0
        self.cum_cap_pos[:] = 0
        self.cum_cap_twins[:] = 1.
        self.cap_mine.reset(prior_mean, prior_diff2, num)

    def set_delta(self, d):
        self.threshold = 1 / d
        self.cap_mine.set_delta(d)

    def set_cand(self, cand):
        pass

    def last_sample(self):
        return self.samples[self.s_ptr-1]

    def add_sample(self, samples: float32[:]):
        sample = samples[0]
        self.samples[self.s_ptr] = sample
        self.capitals[self.s_ptr] = self.cap_mine.advance(sample)
        self.s_ptr += 1

    def advance(self, samples: float32[:], m_lst: float32[:]):
        x = samples[0]
        lbd = self.cap_mine.advance(x)
        cum_capitals = self.cum_cap_twins
        trunc_scale = self.trunc_scale
        for mi, m in enumerate(m_lst):
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


if __name__ == '__main__':
    for _ in range(10):
        X = np.random.binomial(1, 0.72653, 40000).astype(np.float32)
        # mu, num = vanilla_betting(X, .01, 1000, .05, .05, .5, .25, 1)
        # print(mu, num)