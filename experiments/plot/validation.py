import argparse

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sn

from experiments.constants import methods, epsilons, deltas, sels
from experiments.plot.helper import boxplot
import experiments.set_plot

method_legend = [f'{b.title()}{c.title()}' for b, c in methods]
method_legend[1], method_legend[0] = method_legend[0], method_legend[1]
method_legend[2], method_legend[3] = method_legend[3], method_legend[2]
line_styles = ['-', '--', (0, (3, 1, 1, 1)), ':']
def swap(array):
    array[:] = array[[1,0, 3, 2]]

def prepare(x):
    return np.repeat(np.asarray(x)[:, None], len(methods), axis=1)

def multi_plot(x, y, ax, legend=True, log=True, title=''):
    ax.plot(prepare(x), y.T)
    for line, ls in zip(ax.get_lines(), line_styles):
        line.set_linestyle(ls)
    # ax.set_title(title)
    if log:
        ax.set_yscale('log')
        ax.set_xscale('log')
    if legend:
        ax.legend(method_legend,labelspacing=0.1, handlelength=.8,
                  columnspacing=0.2,
              bbox_to_anchor=(-0.03, -.11),
                  loc='lower left',
              fancybox=False, frameon=False, ncol=2, handletextpad=0.1)


def num_time_plot(x, y, axs, log=True, title=''):
    # fig, ax = plt.subplots(2, 1, sharex=True)
    x_sp = 1.04
    multi_plot(x, np.mean(y[..., 0], axis=2), axs[0], True, log=log)
    axs[0].text(x_sp, 0.5, '(b)', transform=axs[0].transAxes)
    multi_plot(x, np.mean(y[..., 1], axis=2), axs[1], False, log=log)
    axs[1].text(x_sp, 0.5, '(c)', transform=axs[1].transAxes)
    # plt.show()

def mu_box_plot(label, data, axs, xlog=False):
    # label =[f'{l:.3f}'.rstrip('0')[1:] for l in label]
    # fig, ax = plt.subplots(1, 4, sharey=True)
    for i, method in enumerate(method_legend):
        axs[i].set_title(method, y=1., pad=-8)
        if xlog:
            axs[i].set_xscale('log')
        boxplot(axs[i], data[i], label)
        axs[i].axhline(y=0.71413, linestyle='--')
    # fig.suptitle(title)
    # plt.show()

def run_time_fix_num():
    from experiments.fix_time import scale
    data = np.load('resource/run_time_fix_num.npy')
    data = np.mean(data, axis=2)

    fig, ax = plt.subplots()
    ax.plot(prepare(scale), data.T)
    # ax.set_title('running time for sampling')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(method_legend)
    plt.show()


def full_by_pos(cap_pos, color, pict):
    # cap_pos = np.max(cap_pos, axis=1)
    for i in range(1000):
        pict[i][:cap_pos[i]] = color


def final_fig(pos, bo, df):
    plt.rcParams["figure.figsize"] = (6, 1.5)
    mpl.rcParams['hatch.linewidth'] *= 0.5
    fig, ax = plt.subplots(1, 3, width_ratios=[1/4, 1/4, 1/2])
    compute_reduce(pos, bo, ax[:2])
    compare_bar(df, '# of total algorithms', ax[-1], 'brief', 'run time (s)',
                'method', [h*2 for h in ['\\', '+', 'x']])
    ax[-1].set_xlabel('# of total algorithms\n(c)')
    sn.move_legend(ax[-1], ncol=2, frameon=False, fancybox=False,
                   loc='upper left', handlelength=1.2, labelspacing=0.5, columnspacing=0.2)
    plt.show()

def compute_reduce(pos, bo, ax):
    img = np.zeros((1000, 2400))
    full_by_pos(pos[1], 10, img)
    full_by_pos(pos[0], 20, img)
    img[0, 0] = 14
    ax[1].imshow(img, aspect='auto', cmap='YlOrBr', origin='lower',
              extent=(0., 2400., 0., 1000.))
    x = np.arange(len(bo[0]))
    ax[1].plot(x, bo[0], x, bo[1], dashes=[2,1], color='blue')


    data2 = np.load('resource/control_m.npy').T * 1000
    img2 = np.zeros((1000, 3200))
    img2[:, :data2.shape[1]] = 6
    img2[0, 0] = 20
    ax[0].imshow(img2,aspect='auto', cmap='YlOrBr', origin='lower',
              extent=(0., 3200., 0., 1000.))
    data2[0] -= 1
    data2[1] += 1
    data2 = np.clip(data2, 0., 1000.)
    x = np.arange(data2.shape[1])
    ax[0].plot(x, data2[0], x, data2[1], dashes=[2,1], color='blue')

    ticks = np.linspace(0, 1000, 6)
    tlabels = [f'{t/1000.:.1f}' for t in ticks]
    ax[0].set_yticks(ticks)
    ax[0].set_yticklabels(tlabels)
    ax[1].set_yticklabels(tlabels)
    ax[0].set_ylabel('$m$')
    ax[1].set_ylabel('$m$')
    for i in range(2):
        ax[i].set_xlabel(f'# of samples\n ({"b" if i else "a"})')

def box_and_num_time(x, a, name, xlog=False, ablog=True):
    plt.rcParams["figure.figsize"] = (6, 3)

    swap(a)
    figs = plt.figure().subfigures(1, 2, width_ratios=[5/8, 3/8])
    axbox = figs[0].subplots(2, 2, sharey=True, sharex=True)
    axnt = figs[1].subplots(2, 1, sharex=True)
    for i in range(2):
        axbox[i, 0].set_ylabel(r'$\widehat{sel}$')
    figs[0].text(0.5, 0.01, f'{name}\n(a)')
    mu_box_plot(x, a[..., 0], axbox.flat, xlog)
    num_time_plot(x, a[..., 1:], axnt, log=ablog)
    axnt[0].set_ylabel('# of samples')
    axnt[1].set_ylabel('run time (s)')
    axnt[1].set_xlabel(name)
    plt.subplots_adjust(top=1.0, bottom=0.125, left=0.07,
                        right=0.92, hspace=0.0, wspace=0.0)
    plt.show()

def onebox_and_num_time(x, a, name):
    plt.rcParams["figure.figsize"] = (6, 2)
    labelpad = -1
    swap(a)
    figs, axs = plt.subplots(1, 3, width_ratios=[3/7, 2/7, 2/7])
    box = axs[0]
    box.set_title(method_legend[-1])
    box.set_ylabel(r'normalized $\widehat{sel}$', labelpad=-5)
    box.set_xlabel('true sel.\n(a)')
    boxplot(box, a[-1, ..., 0], x)
    numplot = axs[1]
    numplot.set_xlabel(name)
    multi_plot(x, np.mean(a[..., 1], axis=2), numplot, False, log=False)
    numplot.set_ylabel('# of samples',labelpad=labelpad)
    # numplot.set_yticklabels(numplot.get_yticklabels(),rotation=-60, ha='right')
    numplot.legend(method_legend, labelspacing=.1, handlelength=1,
                   columnspacing=.5, fancybox=False, frameon=False,
                   loc='upper center', bbox_to_anchor=(1.2, 1.3), ncol=len(
            method_legend))
    numplot.set_xlabel('true sel.\n(b)')
    time = axs[2]
    multi_plot(x, np.mean(a[..., -1], axis=2), time, False, log=False)
    time.set_ylabel('run time (s)', labelpad=labelpad)
    time.set_yscale('log')
    time.set_xlabel('true sel.\n(c)')
    # figs.text(0.5, 0.07, f'true selectivity')
    plt.show()

def compare_bar(df, name, ax, legend, y_label, hue_name, hatches):
    g = sn.barplot(df, ax=ax, errorbar=('ci', False), x=name,
                   y=y_label, width=0.8, edgecolor='k',
                   hue=hue_name, legend=legend)
    for h, hues in zip(hatches, g.containers):
        for bar in hues:
            bar.set_hatch(h)
    if legend:
        leg = g.get_legend()
        leg.set_title(None)
        for hatch, handle in zip(hatches, leg.legend_handles):
            handle.set_hatch(hatch)
    # ax.set_xscale('log')

def compare_all(eps, dels, sel, axes, first_legend, y_label):
    import pandas as pd
    mpl.rcParams['hatch.linewidth'] = 0.5
    hatches = [h*6 for h in ['-', '+', '\\', 'x']]
    hue_name = 'method'
    group = (eps, dels, sel)
    for i in range(3):
        a, label, name = group[i]
        da = {f'{l:.3f}'.rstrip('0')[1:]: d for l, d in zip(label, a)}
        df = pd.DataFrame(da, index=method_legend)
        df.index.name = hue_name
        df = df.reset_index().melt(id_vars=hue_name, var_name=name,
                                   value_name=y_label)
        compare_bar(df, name, axes[i], 'brief' if first_legend and i == 0
        else False, y_label, hue_name, hatches)

def reduce(data, mean, eps):
    return np.sum((np.abs(data - mean)) > eps, axis=-1)

def diff_mu_sam(orig, cv, mean, epsilon, scale=1):
    from collections.abc import Sequence
    diff = []
    cur_mean = mean if isinstance(mean, Sequence) else [mean, mean]
    od = reduce(orig[..., 0], cur_mean[0], epsilon)
    cvd = reduce(cv[..., 0], cur_mean[1], epsilon)
    dist = (od - cvd) / scale
    swap(dist)
    diff.append(dist.T)

    od = np.mean(orig[..., 1], axis=-1)
    cvd = np.mean(cv[..., 1], axis=-1)
    dist = (od - cvd) / od
    swap(dist)
    diff.append(dist.T)
    return diff

def plot_comp(eps, delta, sel, ):
    plt.rcParams["figure.figsize"] = (6, 2)
    fig, axes = plt.subplots(2, 3)
    ylable=['$\Delta_e$', '$\Delta_s$']
    for i in range(2):
        compare_all((eps[i], epsilons, '$\epsilon$\n(a)'),
                    (delta[i], deltas, '$\delta$\n(b)'),
                    (sel[i], np.linspace(0.1, 0.9, 9), 'true '
                                                            'sel.\n(c)'),
                    axes[i], first_legend=True if i == 0 else False,
                    y_label=ylable[i])
        for j in range(2):
            axes[i, j].set_xticklabels(axes[i, j].get_xticklabels(),
                                       rotation=45, ha='right')

    sn.move_legend(axes[0, 0], labelspacing=.1, handlelength=1,
                   columnspacing=.5, fancybox=False, frameon=False,
                   loc='upper center', bbox_to_anchor=(1.8, 1.3), ncol=len(
            method_legend))
    axes[0, 0].set_yscale('log')
    axes[-1, -1].set_xlabel('true sel.\n(c)', labelpad=5)
    for i in range(3):
        axes[0, i].get_xaxis().set_visible(False)

    for i in range(1, 3):
        for j in range(2):
            axes[j, i].set_ylabel('')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--do', '-d', type=str,
                        required=False, default='veps',
                        choices=['veps', 'vdelt', 'comd', 'vsel', 'cv'])
    args = parser.parse_args()
    match args.do:
        case 'veps':
            a = np.load('resource/multi_epsilon.npy')
            box_and_num_time(epsilons, a, r'$\epsilon$', True, True)
        case 'vdelt':
            a = np.load('resource/delta_eps0.01.npy')
            box_and_num_time(deltas, a, r'$\delta$')
        case 'comd':
            input = np.load('resource/cs.npz')
            df_cmp = pd.read_pickle('resource/runtime_cmp.pkl')
            final_fig(input['pos'], input['bo'], df_cmp)
        case 'vsel':
            input = np.load('resource/multi_sel.npy')
            input[..., 0] -= np.array(sels)[None, :, None]
            onebox_and_num_time(sels, input, r'true sel.')
        case 'cv':
            a = np.load('resource/multi_epsilon.npy')
            cv = np.load('resource/cv_eps.npy')
            diff_eps = diff_mu_sam(a, cv, 0.71413, epsilons[None, :, None], 50)

            a = np.load('resource/delta_eps0.01.npy')
            cv = np.load('resource/cv_del.npy')
            diff_del = diff_mu_sam(a, cv, 0.71413, .01,  (deltas * 1000)[
                                                         None,:])

            a = np.load('resource/multi_sel.npy')[:, 2:-1:2, ...]
            sel_s = np.array(sels)[2:-1:2]
            cv = np.load('resource/cv_sel.npy')
            cv_sel = np.array([.1110, .1996, .2975, .4151, .5180, .6062, .7141,
            .7735, .9172])
            diff_sel = diff_mu_sam(a, cv, [sel_s[None,:, None],
                                           cv_sel[None, :, None]], .01, 50)

            plot_comp(diff_eps, diff_del, diff_sel)

            # data = np.load('resource/cv_sel.npy')
            # sels =[.1110, .1996, .2975, .4151, .5180, .6062, .7141, .7735, .9172]
            # data[..., 0] -= np.array(sels)[None,:, None]
            # onebox_and_num_time(sels, data, r'true sel.')

        case _:
            raise ValueError




