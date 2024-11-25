import argparse

import numpy as np
from matplotlib import pyplot as plt

from experiments.constants import sample_nums
from experiments.plot.helper import boxplot
import experiments.set_plot
from experiments.plot.viva_breakdown import break_down_plot, weight_counts


def various_sample_num(data, label):
    plt.rcParams["figure.figsize"] = (6, 2)
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    boxplot(ax, data, label)
    ax.set_xlabel('sample sizes')
    ax.set_ylabel('estimated selectivity')
    # ax.set_title('Distribution of Estimated Selectivity Across Various Sample Sizes')
    # plt.axhline(y=0.71413, linestyle='--')
    plt.show()

def plot_sample(label):
    plt.rcParams["figure.figsize"] = (6, 2)
    data = np.squeeze(np.load('resource/sam_num.npy'))
    fig, axs = plt.subplots(1, 3, width_ratios=[3/8, 3/8, 1/4])
    ax = axs[0]
    ax.set_xscale('log')
    boxplot(ax, data, label)
    ax.set_xlabel('sample sizes\n(a)')
    ax.set_ylabel(r'$\widehat{sel}$')
    ax.axhline(y=0.71413, linestyle='--')
    # ax.set_title('Distribution of the Estimated Selectivity')

    x, y = np.load('resource/increase_models.npy')
    ax = axs[2]
    ax.plot(x, y)
    ax.set_xlabel('# of total algorithms\n(c)')
    ax.set_ylabel('run time (s)')
    # ax.set_title('Selectivity Estimation Cost')
    break_down_plot(weight_counts, 0.7, axs[1], '\n(b)')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pick', '-p', type=str,
                        required=False, default='vnum',
                        choices=['vnum', 'distmodels'])
    args = parser.parse_args()
    match args.pick:
        case 'vnum':
            various_sample_num(np.load('resource/sam_num.npy'), sample_nums)
        case 'distmodels':
            plot_sample(sample_nums)
        case _:
            raise ValueError
