from math import log

from numba import float32, njit


@njit((float32,))
def cal_c(delta):
    return 2*log(2 / delta)