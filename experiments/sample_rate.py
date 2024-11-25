import random

import numpy as np

from experiments.constants import sample_nums
from experiments.utilities import load_data


def various_sample_rates(data, nums, repeat=1000, suffix=''):
    m_ctn = []
    for md in data:
        re_ctn = []
        for _ in range(repeat):
            shuffled = np.random.permutation(md)
            rand = random.randint(-15, 15)
            rat_ctn = [np.sum( s := shuffled[:round(num) + rand])/ len(s)
                       for num in nums]
            re_ctn.append(rat_ctn)
        m_ctn.append(re_ctn)
    np.save(f'sam_num{suffix}', np.squeeze(np.array(m_ctn)))

if __name__ == '__main__':
    # pure sampling
    # rates = np.array([0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5])
    # various_sample_rates([load_data('../pseudo_label.pkl', 'm')],
    #                      sample_nums, 10000)
    various_sample_rates([load_data('../pseudo_label.pkl', 'm')], [485],
                         10000, suffix='m0.0033')
