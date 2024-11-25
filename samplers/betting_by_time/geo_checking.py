import math

import numpy as np
from numba import float32, int32, njit, float64, config
from numba.experimental import jitclass

from samplers.betting_by_time.utilities import cal_c

use_numba = True

config.DISABLE_JIT = not use_numba

@jitclass
class TimeSliceOptLambda:
    if use_numba:
        cum: float32
        cum_diff2: float32
        c: float32
        lbd_cum: float32
        lbd2_cum: float32
        t: float32

    def __init__(self, prior_mean: float32=0.5, prior_var: float32=0.25,
                 num: int32=1, delta: float32=0.05):
        self.cum = prior_mean * num
        self.cum_diff2 = prior_var * num
        self.c = cal_c(delta)
        lbd = math.sqrt(self.c/self.cum_diff2)
        self.lbd_cum = lbd * num
        self.lbd2_cum = self.lbd_cum * lbd
        self.t = num - 1

    def reset(self, pior_mean, pior, num):
        self.cum_diff2 = pior * num
        lbd = math.sqrt(self.c / self.cum_diff2)
        self.lbd_cum = lbd * num
        self.lbd2_cum = self.lbd_cum * lbd
        self.t = num - 1
        self.cum = pior_mean * num

    def set_delta(self, d):
        self.c = cal_c(d)

    def _update(self, lbd: float32, n: int32):
        self.lbd_cum += lbd * n
        self.lbd2_cum += lbd * lbd * n


    def advance(self, pool: float32[:], cand_num):
        n = np.sum(pool[cand_num:])
        self.t += n
        t = self.t
        self.cum += np.sum(pool[:cand_num] * pool[cand_num:])
        self.cum_diff2 += np.sum(np.square(pool[:cand_num] - self.cum / t) *
                                pool[cand_num:])

        # lambda* see the paper
        sigma2 = self.cum_diff2 / t
        a = sigma2 * self.lbd2_cum + self.c
        b = self.lbd_cum / n
        # lbd = math.sqrt(self.c/(sigma2*n))
        lbd = math.sqrt(b*b + a / (sigma2*n)) - b
        # update
        self._update(lbd, n)
        return lbd

@njit((float32[:], float32[:], int32), cache=True)
def update_pool(pool, samples, cand_num):
    pool[cand_num:] = np.count_nonzero(pool[:cand_num, np.newaxis] ==
                                                      samples, axis=1)

@njit((float32, float32, float32[:], float64, float32), fastmath=True, cache=True)
def geo_single_bet_on(trunc_scale, m: float32, spool: float32[:], cum_cap: float64,
                      capital: float32):
    if cum_cap < 1e-16:
        # inf * 0 = 0. cum_capital unchanged
        return cum_cap
    size = len(spool)//2
    for i in range(size):
        n = spool[i + size]
        if n == 0.:
            continue
        bet_err = spool[i] - m
        if abs(bet_err) > 1e-9:
            #  x == m. Use convention that inf * 0 = 0. We still have
            # a martingale under the null. Thus, 1 + 0 = 1, cum_capital unchanged
            cum_cap *= (1 + bet_err *
                        min(max(capital,-trunc_scale / (1. + 1e-9 - m)),
                            trunc_scale / (m + 1e-9))
                        ) ** n
    return cum_cap

@jitclass
class GeoCheckingCapital:
    if use_numba:
        cum_cap_twins: float64[:, :]
        cum_cap_pos: int32[:, :]
        capitals: float32[:]
        samples: float32[:, :]
        trunc_scale: float32
        threshold: float32
        grid_num: int32
        s_ptr: int32
        cap_mine: TimeSliceOptLambda
        cand_num: int

    def __init__(self, delta=.05, trunc_scale=.5, grid_num=1000,
                 prior_mean=0.5, prior_var=0.25, num=1, cand_num=2):
        # twin[0] means we bet on m < mean, twin[1] means we bet on m > mean
        self.cum_cap_twins = np.ones((grid_num + 1, 2))
        self.cum_cap_pos = np.zeros((grid_num + 1, 2), dtype=np.int32)
        self.capitals = np.zeros(1000, dtype=np.float32)
        self.trunc_scale = trunc_scale
        self.threshold = 1 / delta
        self.grid_num = grid_num
        self.samples = np.zeros((1000, cand_num*2), dtype=np.float32)
        if cand_num == 2:
            cand = np.array([0., 1.])
            self.samples[:, :cand_num] =  cand
        self.s_ptr = 0
        self.cap_mine = TimeSliceOptLambda(prior_mean, prior_var, num,
                                           delta * 0.5)
        self.cand_num = cand_num

    def reset(self, prior_mean=0.5, prior_var=0.25, num=1):
        self.samples[:self.s_ptr, self.cand_num:] = 0.
        self.cum_cap_pos[:] = 0
        self.cum_cap_twins[:] = 1
        self.capitals[:] = 0
        self.s_ptr = 0
        self.cap_mine.reset(prior_mean, prior_var, num)

    def set_cand(self, cand):
        self.cand_num = len(cand)
        if self.samples.shape[1] != self.cand_num*2:
            self.samples = np.zeros((1000, self.cand_num*2), dtype=np.float32)
        self.samples[:, :self.cand_num] = cand

    def set_delta(self, delta):
        self.threshold = 1 / delta
        self.cap_mine.set_delta(delta)


    def add_sample(self, samples: float32[:]):
        pool = self.samples[self.s_ptr]
        update_pool(pool, samples, self.cand_num)
        self.capitals[self.s_ptr] = self.cap_mine.advance(pool, self.cand_num)
        self.s_ptr += 1

    def last_sample(self)->float32[:]:
        return self.samples[self.s_ptr-1]

    def lat_capital(self):
        return self.capitals[self.s_ptr-1]

    def advance(self, samples: float32[:], m_lst:float32[:]):
        self.add_sample(samples)
        capital = self.lat_capital()
        cum_capitals = self.cum_cap_twins
        sample = self.last_sample()
        trunc_scale = self.trunc_scale
        for mi, m in enumerate(m_lst):
            cum_cap_twin = cum_capitals[mi]
            cum_cap_twin[0] = geo_single_bet_on(trunc_scale, m,
                                                sample,
                                               cum_cap_twin[0], capital)
            cum_cap_twin[1] = geo_single_bet_on(trunc_scale, m,
                                                sample,
                                                cum_cap_twin[1], -capital)


if __name__ == '__main__':
    times = [a for i in range(10, 1000) if 5000 <= (a := math.ceil(1.04**i))
             <= 41000]
    times.insert(0, 0)
    times = np.array(times, dtype=np.int32)
    repeat = 100
    count = 0
    error = 0
    mean = 0.74353
    # for i in range(repeat):
    #     x = np.random.binomial(1, mean, times[-1]+1).astype(np.float32)
    #     # mu, num = betting_fast(x, times, .01, 1000, .5, .5, .05, .25, 1)
    #     count += num
    #     if abs(mean - mu) > .01:
    #         error += 1
    # print('error:', error, 'avg num:', count/repeat)