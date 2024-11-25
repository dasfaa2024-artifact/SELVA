import matplotlib.pyplot as plt
import numpy as np
import matplotlib

plt.style.use('seaborn-v0_8-paper')
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams["figure.figsize"] = (6, 2.2)

species = (
    "News",
    "Traffic",
    "Sport",
    "Bias"
)

weight_counts = {
    "ingest video": np.asarray([40.26, 25.66, 34, 27.1]),
    "estimate acc.": np.asarray([86.7, 27.4, .0, 15]),
    "estimate sel.": np.asarray([138.5, 98.6, 39, 96.5]),
    "exec. query": np.asarray([109.1, 103.56, 77.5, 86.15]),
}
weight_counts_10 = {
    "Ingest Video": np.asarray([40.26, 25.4, 34, 27.5]),
    "F1 Score Profiling": np.asarray([345, 42, 0, 30]),
    "Selectivity Profiling": np.asarray([200.5, 95, 39, 96.5]),
    "Query Exec.": np.asarray([109.1, 104.07, 77.5, 86.24]),
}


def text_color(face_color):
    r, g, b, _ = face_color
    return 'white' if r * g * b < 0.5 else 'darkgrey'

def break_down_plot(costs_m, width, ax, xlabel_suffix=''):
    import pandas as pd
    df = pd.DataFrame(costs_m, index=species)
    # df = df.melt(id_vars='dataset')

    # fig, axes = plt.subplots(1, 2)
    ax = df.plot(stacked=True, kind='bar', rot='horizontal', ax=ax,
                 width=width, edgecolor='k')
    bars = ax.patches
    hatches = [h * 4 for h in '+/x\\' for _ in range(4)]
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    # for rect in ax.patches:
    #     height = rect.get_height()
    #     if height == 0:
    #         continue
    #
    #     x = rect.get_x()
    #     y = rect.get_y()
    #
    #     # The height of the bar is the data value and can be used as the label
    #     label_text = f'{height:.1f}'  # f'{height:.2f}' to format decimal values
    #
    #     # ax.text(x, y, text)
    #     label_x = x + width / 2
    #     label_y = y + height / 2
    #     ax.text(label_x, label_y, label_text, ha='center', va='center', c=text_color(rect.get_facecolor()), fontsize='small')

    ax.set_ylabel('run time (s)')
    ax.set_xlabel('queries'+xlabel_suffix)
    ax.legend(labelspacing=0.1, handlelength=1.2, columnspacing=0.6,
              bbox_to_anchor=(.5, 1.13), loc='center',
              fancybox=False, frameon=False, ncol=2, handletextpad=0.2)

    # the second subplot
    # cat_color = [p.get_facecolor() for p in ax.patches][::len(species)]
    # cat_names = list(costs_m.keys())
    # labels = species[:]
    # data = np.array(list(costs_m.values())).T
    # data /= data.sum(axis=1).reshape(-1, 1)
    # data_cum = data.cumsum(axis=1)
    # ax = axes[1]
    # # print(data)
    # ax.invert_yaxis()
    # ax.xaxis.set_visible(False)
    # ax.set_xlim(0, np.sum(data, axis=1).max())
    # for i, (colname, color) in enumerate(zip(cat_names, cat_color)):
    #     widths = data[:, i]
    #     starts = data_cum[:, i] - widths
    #     rects = ax.barh(labels, widths, left=starts, height=width, label=colname, color=color)
    #     bar_label = [f'{round(d*100-0.08)}%' if d > .0 else '' for d in widths]
    #     ax.bar_label(rects, bar_label, label_type='center', color=text_color(color), fontsize='small')
    # ax.legend(ncols=len(cat_names), bbox_to_anchor=(-0.1, 1), loc='lower left', fontsize='x-small')
    plt.show()



if __name__ == "__main__":
    fig, axs = plt.figure()
    break_down_plot(weight_counts, 0.35, axs)
