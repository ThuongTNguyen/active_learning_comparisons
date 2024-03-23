from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns; sns.set()

def get_plot_template(title=None, return_name_position_map=False):
    """
    Takes some optional arguments.
    :param title: This is the figure suptitle. The caller might add this too, but can be finicky, so you can pass
        it here, and this function return a figure with the suptitle added and adjusted.
    :param return_name_position_map: dict, where keys are tuples (clf, rep) and values are the index where a plot
        corresponding to a particular clf and rep should appear. This index is to be used in "axes", on of the return
        values.
    :return:
    """
    annot_background_color = '#e3c565'
    annot_font_size = 30

    fig = plt.figure(figsize=(32, 22), constrained_layout=True)

    widths = [0.15, 1, 1, 1]
    heights = [0.1, 1, 1, 0.05, 1]
    spec = fig.add_gridspec(ncols=4, nrows=5, width_ratios=widths,
                              height_ratios=heights)
    axes = dict()
    count = -1

    # top most row is for col labels
    for i, j, textstr in [(0, 1, 'wordvecs'), (0, 2, 'USE'), (0, 3, 'MPNet')]:
        ax = fig.add_subplot(spec[i, j])
        ax.axis("off")
        ax.annotate(textstr, (0.35, 0.3), xycoords='axes fraction', va='center',
                    backgroundcolor=annot_background_color, fontsize=annot_font_size)

    # left most col is for row labels
    for i, j, textstr in [(1, 0, 'LinearSVC'), (2, 0, 'RF')]:
        ax = fig.add_subplot(spec[i, j])
        ax.axis("off")
        ax.annotate(textstr, (0.5, 0.5), xycoords='axes fraction', va='center', backgroundcolor=annot_background_color,
                    fontsize=annot_font_size, rotation=90)

    for row in range(1, 3):
        for col in range(1, 4):
            count += 1
            ax = fig.add_subplot(spec[row, col])
            axes[count] = ax

    # draw a line to separate out BERT
    ax = fig.add_subplot(spec[3, :])
    ax.axis("off")
    ax.axhline(y=0.5, c='#E97451', linestyle="-", linewidth=2)

    # the BERT plot - the annot. is actually in the left  neighboring plot
    ax = fig.add_subplot(spec[4, 2])
    axes[max(axes.keys()) + 1] = ax
    # annot. for BERT plot
    ax = fig.add_subplot(spec[4, 1])
    ax.axis("off")
    ax.annotate('RoBERTa', (0.7, 0.9), xycoords='axes fraction', va='center', backgroundcolor=annot_background_color,
                fontsize=annot_font_size, rotation=0)
    fig.subplots_adjust(hspace=0.24)
    # plt.savefig(f'generated/gridspec.png')
    if title:
        fig.suptitle(title, fontsize=annot_font_size + 4, y=0.94)
    name_position_map = {('BERT', 'BERT'): 6,
                         ('LinearSVC', 'wordvecs'): 0,
                         ('LinearSVC', 'USE'): 1,
                         ('LinearSVC', 'MPNet'): 2,
                         ('RF', 'wordvecs'): 3,
                         ('RF', 'USE'): 4,
                         ('RF', 'MPNet'): 5}

    if return_name_position_map:
        return fig, axes, name_position_map
    else:
        return fig, axes

def demo_usage_plot_template():

    fig, axes = get_plot_template(title="random plots")
    # we'll randomly pick functions from here to plot
    funcs = [np.sin, np.square, np.exp, np.absolute]
    for k, ax in axes.items():
        print(f"Plotting for axis={k}.")
        x = np.linspace(0, 20, 100)
        fn = funcs[np.random.choice(len(funcs))]
        y = fn(x)
        ax.plot(x, y, label=f"{fn.__name__}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        # ax.set_title(f"Plot {k}.")
        ax.legend()

    plt.savefig(f'scratch/gridspec.png', bbox_inches='tight')

if __name__ =='__main__':
    demo_usage_plot_template()