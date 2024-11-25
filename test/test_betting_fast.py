import math
from itertools import pairwise, product

import numpy as np
import pytest

from samplers import update_pool, LambdaBet, CSBetting, advance_betting_s1, \
    advance_betting_s2_down, betting_fast
from samplers import predmix_eb

class MockLbd:
    def __init__(self, lbds):
        self.lbds_ = lbds
        self.ptr = 0
    def advance(self, pool):
        self.ptr += 1

    def lbds(self):
        return self.lbds_[:self.ptr]

def bisamples(pool):
    out = []
    for i in range(2):
        out.extend([pool[i]]*int(pool[i+2]))
    return np.random.permutation(out)

pools = np.array([(.345, .44, 39, 21), (.345, .44, 0, 60), (.345, .44, 60, 0)])

@pytest.mark.parametrize('pool', pools)
def test_update_pool(pool):
    bis = bisamples(pool)
    update_pool(pool, bis)
    assert pytest.approx(np.dot(pool[:2], pool[2:])) == np.sum(bis)

def test_ad_lambda():
    times = [a for i in range(10, 1000) if (a := math.ceil(1.5 ** i)) <= 1000]
    times.insert(0, 0)
    X = np.random.binomial(1, 0.5, times[-1] + 1).astype(np.float32)
    lbds_gen = LambdaBet()
    lbds_true = predmix_eb(X, times)
    pool = np.empty(4)
    for s, e in pairwise(times):
        update_pool(pool, X[s:e])
        lbds_gen.advance(pool)
    assert np.isclose(np.repeat(lbds_gen.lbds(), np.diff(times)),
                      lbds_true, 0,
                      0.02).all()

@pytest.fixture(name='cum_capital')
def fixture_cum_cap():
    return [[10., 20.], [5., 0.], [0., 8.]]
@pytest.fixture(name='capital_lbds')
def fixture_capital_lbds():
    return [-2.1, -.8, -.2, .4, 1.4, 3.2]
m = 0.34
mi = int(m*1000)
out_lbds = [(-0.7575757564279155, 1.4705882309688578),
            (-0.7575757564279155, .8),
            (-.2, .2), (.4, -.4),
            (1.4, -0.7575757564279155),
            (1.4705882309688578, -0.7575757564279155)]
@pytest.fixture(name='true_mean')
def fixture_true_mean():
    mean = np.sum(pools[:, :2] * pools[:, 2:]) / np.sum(pools[:, 2:])
    return mean

@pytest.fixture(name='gambler')
def fixture_gambler(capital_lbds):
    return CSBetting(MockLbd(capital_lbds))

class TestCSBetting:
    def test_bet_once(self, cum_capital, capital_lbds, gambler):
        pool = pools[0]
        for cum_cap, lbd, both in product(cum_capital, capital_lbds, [2, 0]):
            out = list(cum_cap)
            gambler.once_bet_on(m, pool, out, lbd, both)
            idx = capital_lbds.index(lbd)
            gt = list(cum_cap)
            for i in range(2):
                for j in range(2):
                    bet_err = pool[j] - m
                    gt[i] *= (1+out_lbds[idx][i] *bet_err)**pool[j+2]
            assert pytest.approx(out[0]) == gt[0]
            if both:
                assert pytest.approx(out[1]) == gt[1]

    def test_get(self, cum_capital, capital_lbds, gambler):
        for pool in pools:
            gambler.add_sample(bisamples(pool))

        assert not gambler.cum_cap_pos[mi]
        gambler.prebet(m, mi, capital_lbds[:len(pools)])
        assert gambler.cum_cap_pos[mi]
        assert gambler.cum_cap_twins[mi][1] > 40

    def test_probe(self, cum_capital, capital_lbds, gambler):
        samples = bisamples(pools[0])
        gambler.add_sample(samples)
        cum_pool = np.ones(2)
        assert gambler.probe(m, cum_pool, capital_lbds[0], samples)
        gambler.prebet(m, mi, capital_lbds[:1])
        assert np.isclose(cum_pool, gambler.cum_cap_twins[mi]).all()

    def test_when_touch_at(self, gambler):
        cum_cap =  32
        lbd = 0.1
        samples = np.array([1] * 50)
        gambler.add_sample(samples)
        assert (math.ceil(math.log(40 / cum_cap, 0.5 * lbd + 1)) ==
                gambler.when_touch_at(cum_cap, samples, lbd, 0.5))

    def test_bet_on(self, capital_lbds, true_mean, gambler):
        for spool in pools:
            gambler.add_sample(bisamples(spool))

        assert not gambler.bet_on(m, mi, capital_lbds[:len(pools)])
        assert gambler.bet_on(true_mean,round(true_mean*1000), capital_lbds[:1])

@pytest.fixture
def lbds():
    return [1.2, 0.9, 0.6]
@pytest.fixture
def samples():
    return bisamples(pools[-1])


class TestBettingState1:
    @pytest.fixture(autouse=True)
    def warm_gambler(self, gambler, lbds, samples):
        gambler.cap_mine = MockLbd(lbds)
        for p in pools[:-1]:
            gambler.add_sample(bisamples(p))

    def test_no_touch(self, gambler, cum_capital, true_mean, samples):
        stride = 0.005
        half_width = 0.02
        l = true_mean - half_width
        u = true_mean + half_width
        le, ue, touch = advance_betting_s1(gambler, stride, samples, l, u)
        assert l == le and u == ue and touch == -1

    def test_touch_u(self, gambler, cum_capital, true_mean, samples):
        stride = 0.005
        width = 0.06
        l = true_mean - 0.02
        u = l + width
        le, ue, touch = advance_betting_s1(gambler, stride, samples, l, u)
        assert l == le and u > ue and touch == 1

    def test_touch_l(self, gambler, cum_capital, true_mean, samples):
        stride = 0.005
        width = 0.06
        u = true_mean + 0.02
        l = u - width
        le, ue, touch = advance_betting_s1(gambler, stride, samples, l, u)
        assert l < le and u == ue and touch == 0

    def test_touch_both(self, gambler, true_mean, samples):
       stride = 0.001
       width = 0.006
       l = true_mean + 0.034
       u = l + width
       le, ue, touch = advance_betting_s1(gambler, stride, samples, l, u)
       assert le < l and ue < u and touch == 1


class TestBettingState2:
    @pytest.fixture(autouse=True)
    def warm_gambler(self, gambler, lbds, samples):
        gambler.cap_mine = MockLbd(lbds)
        for p in pools:
            gambler.add_sample(bisamples(p))
    def test_betting_down(self, gambler, lbds):
        stride = 0.001
        width = 0.006
        u = 0.415
        l = u - width
        samples = bisamples(pools[0])
        gambler.prebet(u, round(u*1000), lbds, len(pools))
        gambler.prebet(l, round(l*1000), lbds, len(pools))
        lbds.append(0.6)
        ue, touch = advance_betting_s2_down(stride,gambler, samples, l, u)
        assert ue < u
@pytest.fixture
def pre_bet(gambler, lbds, samples):
    gambler.cap_mine = MockLbd(lbds)
    lbds.extend(0.6 **(1+.01*i) for i in range(27))
    ctn = []
    times = [0]
    for p in np.repeat(pools, 10, axis=0):
        sam = bisamples(p)
        ctn.append(sam)
        times.append(len(sam) + times[-1])
    return np.random.permutation(np.concatenate(ctn, axis=0)), times

class TestBetting:
    @pytest.mark.parametrize('stride1, stride2', [(.05, .01), (-.05, -.01)])
    def test_betting(self, gambler, pre_bet, true_mean, stride1, stride2):
        prior = ((true_mean + stride1) * 2 - stride2) / 2
        mu = betting_fast(*pre_bet, 0.01, prior, 1000, gambler)
        assert  abs(mu - true_mean) <= 0.01