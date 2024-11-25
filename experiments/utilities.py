from itertools import groupby
import math
from math import log
from random import randint

import numpy as np

from experiments.constants import eps_m, eps_dic, trunc_dic


def gen_times(start, b, end=100000):
    s = math.ceil(log(start, b))
    e = math.ceil(log(end, b))
    times = [math.ceil(b ** i) for i in range(s, e)]
    times = [k for k, _ in groupby(times)]
    times.insert(0, 0)
    times.append(end)
    return np.array(times, dtype=np.int32)

def gen_times_by_epsilon(eps, cap, end=100000):
    if cap == 'geo':
        s, b = eps_m[eps]
        return gen_times(s, b, end)
    else:
        return np.arange(end, dtype=np.int32)

def trunc_for(cap, e):
    return trunc_dic[cap][eps_dic[e]]


def load_data(file, abbr):
    import pandas as pd
    data = pd.read_pickle(file)
    if len(abbr) > 1:
        return [data[m] for m in abbr]
    else:
        return data[abbr]

def gen_samples(samples, repeat, keep=100000):
    return [np.random.choice(samples, keep, replace=False) for _ in range(repeat)]

def is_prime(num, test_count):
    if num == 1:
        return False
    if test_count >= num:
        test_count = num - 1
    for x in range(test_count):
        val = randint(1, num - 1)
        if pow(val, num-1, num) != 1:
            return False
    return True

def generate_big_prime(n=32):
    found_prime = False
    while not found_prime:
        p = randint(2**(n-1), 2**n)
        if is_prime(p, 1000):
            return p