import numpy as np
from numba import float32, njit, int32


__all__ = ['vanilla_betting_factory', 'adaptive_betting_factory']

from samplers.betting_by_time.bet_once import *


@njit((float32[:], float32, float32))
def intersect(a, l, u):
    a[0] = max(a[0], l)
    a[1] = min(a[1], u)

def vanilla_betting(samples, times, prior_mean, delta, grid_num, gambler):
    m_possible = np.linspace(0, 1, grid_num+1).astype(np.float32)
    eps = delta * 2 - 1/grid_num
    cs_bound = np.array([0., 1.], dtype=np.float32)
    threshold = gambler.threshold * 2
    # bound = np.zeros((3102, 2))
    # bound[0] = 0., 1.
    for i in range(len(times)-1):
        gambler.advance(samples[times[i]:times[i+1]], m_possible)
        rang_idx = np.flatnonzero(np.sum(gambler.cum_cap_twins, axis=1) <
                                  threshold)
        intersect(cs_bound, m_possible[rang_idx[0]], m_possible[rang_idx[-1]])
        # bound[i+1] = cs_bound
        if cs_bound[1] - cs_bound[0] < eps:
            break
    return (np.argmin(np.sum(gambler.cum_cap_twins, axis=1)) / grid_num,
            times[i + 1])

def vanilla_betting_factory(cls):
    return njit((float32[:], int32[:], float32, float32, int32,
                 cls.class_type.instance_type))(vanilla_betting)
    # return vanilla_betting

@njit((float32, float32))
def initBound(prior, delta):
    l = min(max(prior-delta, 0.), 1. - delta*2)
    return np.array([l, l+delta*2], dtype=float32)


def adaptive_betting_factory(cls, cap_mtd='geo'):
    bet_on, estimate = bet_on_factory(cls, cap_mtd)

    @njit((cls.class_type.instance_type, float32, float32[:], float32))
    def advance_betting_s1(gambler, scale, bound, win):
        # touch:
        #       -1: no touch, 0: touch l, 1: touch u, 2: touch both
        l, u = bound[0], bound[1]
        touch = -1
        stride = 1/scale
        li, ui = round(l*scale), round(u * scale)
        if bet_on(gambler, l, li, 2):
            touch = 0
        if bet_on(gambler, u, ui, 2):
            touch += 2

        if touch == 2:
            # double-checking, l and u might be both out of CS bound
            cap_twin_l, cap_twin_u = (gambler.cum_cap_twins[li],
                                      gambler.cum_cap_twins[ui])
            which = np.argmax(cap_twin_l)
            if which == np.argmax(cap_twin_u): # overshoot
                touch = which
                width = win + stride # we can slide a window length.
                w_stride = round(width * scale)
                # now slide until the expected bound enters into cs bound
                if which == 1:
                    li = round(l*scale)
                    while l >= 0.:
                        l -= width
                        li -= w_stride
                        if not bet_on(gambler, l, li, 1):
                            if not bet_on(gambler, l, li, 0):
                                touch = 1 # no lower bound found
                            break
                    else:
                        l = 0.
                    u = l + width
                else:
                    ui = round(u*scale)
                    while u <= 1.:
                        u += width
                        ui += w_stride
                        if not bet_on(gambler, u, ui, 0):
                            if not bet_on(gambler, u, ui, 1):
                                touch = 0
                            break
                    else:
                        u = 1.
                    l = u - width

        if touch == 0:
            li, ui = round(l*scale), round(u * scale)
            while li + 1 < ui: # find the new lower bound
                mid = (li + ui) // 2
                if bet_on(gambler, mid*stride, mid, 0):
                    li = mid
                else:
                    ui = mid
            l = ui * stride
            u = l + win
            if u > 1. or bet_on(gambler, u, round(u*scale), 1):
                touch = 2
                u = min(u, 1.)
        elif touch == 1:
            li, ui = round(l*scale), round(u * scale)
            while li + 1 < ui: # find the new upper bound
                mid = (li + ui) // 2
                if bet_on(gambler, mid*stride, mid, 1):
                    ui = mid
                else:
                    li = mid
            u = li * stride
            l = u - win
            if l < 0 or bet_on(gambler, l, round(l*scale), 0):
                touch = 2
                l = max(l, 0.)
        bound[0], bound[1] = l, u
        return touch

    @njit((float32, cls.class_type.instance_type, float32[:], float32))
    def advance_betting_s2_up(stride, gambler, bound, win):
        u = bound[1]
        ui = round(u / stride)
        # first check if we can be sure that the upper bound(u) > mean
        # if so, we can slide down a window length
        width = win + stride
        i_stride = round(width / stride)
        while u <= 1.:
            if not bet_on(gambler, u, ui, 0):
                l = u - win
                break
            u += width
            ui += i_stride
        else: # beyond 1, no need to search more
            bound[0], bound[1] = u - width, 1.
            return True
        flag = False
        # find the position first stepping int the cs bound.
        mi = round(l / stride)
        while bet_on(gambler, l, mi, 0):
            l += stride
            mi += 1
        u = l + win
        # check upper bound
        if u > 1 or bet_on(gambler, u, round(u /stride), 1):
            flag = True
        bound[0], bound[1] = l, min(u, 1.)
        return flag

    @njit((float32, cls.class_type.instance_type, float32[:], float32))
    def advance_betting_s2_down(stride: float32, gambler, bound, win):
        l = bound[0]
        li = round(l / stride)
        # first check if we can be sure that the low bound(l) > mean
        # if so, we can slide down a window length
        width = win + stride
        i_stride = round(width / stride)
        while l >= 0.:
            if not bet_on(gambler, l, li, 1):
                u = l + win
                break
            l -= width
            li -= i_stride
        else: # below 0, no need to search more
            bound[0], bound[1] = 0., l + width
            return True

        flag = False
        mi = round(u/stride)
        # find the position first stepping int the cs bound.
        while bet_on(gambler, u, mi, 1):
            u -= stride
            mi -= 1
        l = u - win
        if l < 0 or bet_on(gambler, l, round(l/stride), 0) :
            flag = True
        bound[0], bound[1] = max(0., l), u
        return flag

    @njit((float32[:], int32[:], float32, float32, int32, cls.class_type.instance_type),
          locals=dict(stride=float32, win=float32))
    def adaptive_betting(samples, times, prior_mean, epsilon, grid_num, gambler):
        stride = 1/grid_num
        win = epsilon * 2
        e_bound = initBound(prior_mean, epsilon)
        i = 0
        # state 1, found direction
        for i in range(len(times) -1):
            gambler.add_sample(samples[times[i]:times[i+ 1]])
            touch = advance_betting_s1(gambler, grid_num, e_bound, win)
            if touch != -1:
                break
        if touch != 2:
            state2_func = advance_betting_s2_up if touch == 0 else advance_betting_s2_down
            # state 2, found terminal m
            for i in range(i+1, len(times)-1):
                gambler.add_sample(samples[times[i]:times[i + 1]])
                touch2 = state2_func(stride, gambler, e_bound, win)

                if touch2:
                    break
        return estimate(gambler, e_bound[0], e_bound[1]), times[i+1]
    return adaptive_betting