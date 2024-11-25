import numpy as np
from numba import int32, float32, njit

__all__ = ['bet_on_factory']


def bet_on_factory(cls, method='geo'):
    match method:
        case 'geo':
            from samplers.betting_by_time.geo_checking import (
                geo_single_bet_on as single_bet_on)
        case 'seq':
            from samplers.betting_by_time.sequence_checking import (
                seq_single_bet_on as single_bet_on)
        case _:
            raise NotImplementedError(f'no such method: {method} for once_bet.')

    @njit((cls.class_type.instance_type, float32, int32, int32))
    def bet_on(gambler, m: float32, mi: int32, which:
    int32=2):
        cum_cap_twin = gambler.cum_cap_twins[mi]
        capitals = gambler.capitals
        cap_size = gambler.s_ptr
        trunc_scale = gambler.trunc_scale
        samples = gambler.samples
        old_pos = gambler.cum_cap_pos[mi]
        if which == 2:
            threshold = gambler.threshold * 2
            if old_pos[0] != old_pos[1]:
                # advance a lagged behind cap
                idx = 0 if old_pos[0] < old_pos[1] else 1
                fidx = idx ^ 1
                cum_cap = cum_cap_twin[idx]
                for i in range(old_pos[idx], old_pos[fidx]):
                    cum_cap = single_bet_on(trunc_scale, m, samples[i],
                                            cum_cap, capitals[i])
                cum_cap_twin[idx] = cum_cap
            else:
                fidx = 0
            cap_p, cap_n = cum_cap_twin
            for i in range(old_pos[fidx], cap_size):
                cap_p = single_bet_on(trunc_scale, m, samples[i],
                                      cap_p, capitals[i])
                cap_n = single_bet_on(trunc_scale, m, samples[i],
                                      cap_n, -capitals[i])
                if max(cap_p, cap_n) > threshold:
                    break
            cum_cap_twin[0] = cap_p
            cum_cap_twin[1] = cap_n
            old_pos[:] = cap_size
            return max(cap_n, cap_p) > threshold
        else:
            threshold = gambler.threshold
            cum_cap = cum_cap_twin[which]
            for i in range(old_pos[which], cap_size):
                cum_cap = single_bet_on(trunc_scale, m, samples[i], cum_cap,
                                        capitals[i] if which == 0 else -capitals[i])
                if cum_cap > threshold:
                    break
            cum_cap_twin[which] = cum_cap
            old_pos[which] = cap_size
            return cum_cap > threshold

    @njit((cls.class_type.instance_type, float32, float32))
    def estimate(gambler, l: float32, u: float32):
        l = max(0, l)
        u = min(1, u)
        if u < l:
            print('upper bound is greater than lower bound')
            return -1
        stride = 1 / gambler.grid_num
        if u - l < stride:
            print('upper bound is the same as the lower bound')
            return l
        l += stride
        li, ui = round(l * gambler.grid_num), round(u * gambler.grid_num)
        for pi in range(li, ui):
            bet_on(gambler, pi * stride, pi, 2)
        mi = np.argmin(np.sum(gambler.cum_cap_twins[li:ui], axis=1))
        return l + mi * stride
    return bet_on, estimate