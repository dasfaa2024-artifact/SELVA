import numpy as np
import pytest

from samplers import betting_cs
from samplers import LambdaBet, CSBetting
from samplers import lambda_predmix_eb

n = 1000
p = 0.65

@pytest.fixture(name='x')
def fixture_x():
    return np.random.binomial(1, p, n)

@pytest.fixture(name='lbder')
def fixture_lbder():
    return LambdaBet()

@pytest.fixture(name='gambler')
def fixture_gambler(lbder):
    lbder.alpha *= .5
    return CSBetting(lbder, grid_num=50)

def test_advance(x, lbder):
    lbds_true = lambda_predmix_eb(x)
    for s in x: lbder.advance(s)
    lbds_re = np.array(lbder.lbds)
    assert np.isclose(lbds_true, lbds_re).all()

def test_cs_betting(gambler, x):
    m_possible = np.linspace(0, 1, gambler.grid_num+1)
    bound = np.fromiter((gambler.advance(s, m_possible) for s in x),
                        dtype=np.dtype((float, 2)), count=n)
    l, u, mu_hat = betting_cs(x, breaks=50, running_intersection=True)
    assert np.isclose(l, bound[:, 0]).all()
    assert np.isclose(u, bound[:, 1]).all()