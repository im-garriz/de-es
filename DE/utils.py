# Numpy for mathematical expressions in arrays
import numpy as np
# Random distributions
import random
# Chart plotting
import matplotlib.pyplot as plt


def set_random_seed(_seed):
    """
    Sets the random seed for repeatability

    :param _seed: random seed to be set

    :return:
    """
    np.random.seed(_seed)
    random.seed(_seed)


def print_progress_curve(values, labels=None, legend=None, error_bar=None, error_step=10, save=False, save_name="saved",
                         figsize=(10,10), title=None, ax_labels=None, ax_ticks=None, n_gens=None, Fontsize=15):
    """
    Prints a progress curve provided arrays of fitness values

    :param values: array of arrays where each array contains fitness values on each generation (array of arrays for
                   supporting multiple plotting)
    :param labels: label for each plot
    :param legend: plt.axes.legend(loc=) argument for legend plotting
    :param error_bar: array of error values for each generation so as to plot error bars
    :param error_step: every few generations to plot the error bar (plotting all might result in illegible charts)
    :param save: whether to save the plot as png or not
    :param save_name: if save=True, name of the saves file
    :param figsize: size of the figure in matplotlib style
    :param title: title of the chart
    :param ax_labels: labels for each axe
    :param ax_ticks: ticks for each axe
    :param n_gens: number of generations (first n_gens) to plot
    :param Fontsize: size of the font for labels and titles

    :return:
    """
    plt.style.use(['science', 'grid'])

    plt.rcParams.update({'font.size': Fontsize})

    figure, axes = plt.subplots(1, 1, figsize=figsize)

    if n_gens is None:
        n_gens = len(values[0])

    for n, value in enumerate(values):
        if labels is None:
            axes.plot(range(n_gens), value[:n_gens])
        else:
            axes.plot(range(n_gens), value[:n_gens], label=labels[n])

        if error_bar is not None:
            error_x = range(0, n_gens, error_step)
            error_y = [value[idx] for idx in error_x]
            error_bar_plot = [error_bar[idx] for idx in error_x]
            axes.errorbar(error_x, error_y, yerr=error_bar_plot, ecolor="LightCoral", linestyle="None",
                          capsize=5)

    if legend is not None:
        axes.legend(loc=legend, fontsize=Fontsize)

    if title is not None:
        axes.set_title(title, fontsize=Fontsize)

    if ax_labels is not None:
        axes.set_xlabel(ax_labels[0], fontsize=Fontsize)
        axes.set_ylabel(ax_labels[1], fontsize=Fontsize)

    if ax_ticks is not None:
        axes.set_xticks(ax_ticks[0], fontsize=Fontsize)
        axes.set_yticks(ax_ticks[1], fontsize=Fontsize)

    plt.tight_layout()

    if save:
        plt.savefig(save_name, format="png", dpi=300)

    plt.show()
