
import seaborn as sns

def boxplot(ax, data, label, vert=True, seaborn=True):
    flierprops = dict(marker='o', markerfacecolor='k', markersize=1,
                      markeredgecolor=None)
    medianlinepros = dict(linewidth=1.5, linestyle='--', color='k')
    meanlinepros = dict(color='r')
    kwargs = dict(flierprops=flierprops, meanprops=meanlinepros,
                  showmeans=True,meanline=True, medianprops=medianlinepros)
    if seaborn:
        data = {l:d for l, d in zip(label, data)}
        ax = sns.boxplot(data, orient='v' if vert else 'h',
                         native_scale=True, ax=ax, **kwargs)
    else:
        ax = ax.boxplot(data.T, tick_labels=label, vert=vert, **kwargs)
    return ax