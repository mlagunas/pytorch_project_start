import matplotlib
from matplotlib import pyplot as plt
import matplotlib.cm as cmx
import numpy as np


def plot_learning_curve(data):
    """
    Plots a learning curve assuming that data is a dict with pairs key-value where
    value is a list containing all the loss values registered during an epoch
    """

    colors = matplotlib.cm.jet(np.linspace(0, 1, len(data)))

    with plt.style.context('ggplot'):
        ## create the figure where we will plot the learning curve
        fig = plt.figure(figsize=(8, 6))

        ## clear plot from old data
        fig.clf()

        ax = plt.gca()

        ## Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ## get values for plotting
        for ix, (key, iter_data) in enumerate(data.items()):
            iter_data = np.array(iter_data)

            # get x-axis range
            x_range = range(iter_data.shape[0])

            iter_data_avg = iter_data.mean(axis=-1)
            iter_data_std = iter_data.std(axis=-1)

            # plot curves
            plt.plot(x_range, iter_data_avg + iter_data_std, linestyle='-', color=colors[ix], lw=0.5, alpha=0.5)
            plt.plot(x_range, iter_data_avg - iter_data_std, linestyle='-', color=colors[ix], lw=0.5, alpha=0.5)
            plt.plot(x_range, iter_data_avg, linestyle='--', marker='o', color=colors[ix], lw=1, ms=2.5, label=key)

            # # fill error area
            plt.fill_between(x_range, iter_data_avg + iter_data_std, iter_data_avg - iter_data_std,
                             alpha=.13, color=colors[ix])

        fig.legend()
        plt.pause(0.1)
