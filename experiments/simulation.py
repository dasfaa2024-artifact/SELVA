import numpy as np
from itertools import pairwise

from numba import njit, float32, int32, jit

from experiments.utilities import load_data, gen_times_by_epsilon

@njit((float32[:,:], float32[:, :], int32), fastmath=True, cache=True)
def multi_shuffle(in_arr, out, num=40000):
    p = np.random.choice(np.arange(len(in_arr[0])), num, replace=False)
    out[:] = in_arr[:, p]

@njit((float32[:], float32[:], int32[:]), cache=True)
def control_variates_sampling(label, cv, times):
    samples = label.copy()
    for i in range(len(times)-1):
        s, e = times[i], times[i+1]
        cv_now = cv[:e]
        cov = np.cov(label[:e], cv_now)[0, 1]
        cv_var = np.var(cv_now)
        cv_mean = np.mean(cv_now)
        samples[s:e] -= cov/cv_var *(cv[s:e] - cv_mean)
    return samples

def muti_control_variates_sampling(label, cv, times):
    from sklearn.linear_model import LinearRegression
    samples = label.copy()
    model = LinearRegression()
    for s, e in pairwise(times):
        coef = model.fit(cv[:e], label[:e]).coef_
        samples[s:e] -= np.sum(coef * (cv[s:e] - np.mean(cv[:e])), axis=1)
    return samples

@njit((float32[:, :], float32[:]), fastmath=True, cache=True)
def simply_cv_sampling(in_arr, out):
    out[:] = in_arr[0]
    cv = in_arr[1]
    cov = np.cov(out, cv)[0, 1]
    out -= cov/np.var(cv) *(cv - np.mean(cv))

def muti_cv_sampling(in_arr, out):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    out[:] = in_arr[0]
    cv = in_arr[1:]
    coef = model.fit(cv.T, out).coef_
    cv -= np.mean(cv, axis=1)[:, np.newaxis]
    cv *= coef[:, np.newaxis]
    out -= np.sum(cv, axis=0)


@njit((float32[:], float32[:], int32, int32), fastmath=True, cache=True)
def repeat_cv_sampling(label, cv, keep, repeat):
    in_arr = np.empty((2, len(label)), dtype=np.float32)
    in_arr[0] = label
    in_arr[1] = cv
    out = np.empty((repeat, keep), dtype=np.float32)
    shuffled = np.empty((2, keep), dtype=np.float32)
    for i in range(repeat):
        multi_shuffle(in_arr, shuffled, num=keep)
        simply_cv_sampling(shuffled, out[i])
    return out

def repeat_multicv_sampling(label, cv, keep, repeat):
    in_arr = np.empty((len(cv)+1, len(label)), dtype=np.float32)
    in_arr[0] = label
    in_arr[1:] = cv
    out = np.empty((repeat, keep), dtype=np.float32)
    shuffled = np.empty((len(cv)+1, keep), dtype=np.float32)
    for i in range(repeat):
        multi_shuffle(in_arr, shuffled, num=keep)
        muti_cv_sampling(shuffled, out[i])
    return out

if __name__ == '__main__':
    labels = load_data('/home/lg/VDBM/multi_predicate/selectivity_sampling'
                '/pseudo_label.pkl', 'mx')
    samples = [sam.astype(np.float32) for sam in multi_shuffle(labels)]
    times =  gen_times_by_epsilon(0.01,'geo', 40000)
    sammples = control_variates_sampling(samples[0], samples[1], times)