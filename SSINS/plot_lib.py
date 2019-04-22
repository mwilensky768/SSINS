"""
A simple plotting library used by SSINS.Catalog_Plot. Direct interaction with
these functions is unnecessary.
"""

from __future__ import absolute_import, division, print_function

import numpy as np


def image_plot(fig, ax, data, cmap=None, vmin=None, vmax=None, title='',
               xlabel='', ylabel='', midpoint=False, aspect='auto',
               cbar_label=None, xticks=None, yticks=None, log=False,
               xticklabels=None, yticklabels=None, mask_color='white',
               cbar_ticks=None, font_size='medium'):

    """
    Plots 2-d images. Can do a midpoint normalize and log normalize.

    Args:
        fig: The fig object to modify
        ax: The axis object to modify
        data: The data to plot
        cmap: A colormap from matplotlib.cm e.g. cm.coolwarm
        vmin: The minimum value for the colormap
        vmax: The maximum value for the colormap
        title: The title for ax
        xlabel: The horizontal axis label
        ylabel: The vertical axis label
        midpoint: Whether to set the midpoint of the colormap to zero (useful for diverging colormaps)
        aspect: Adjusts aspect ratio, see ax.imshow documentation in Matplotlib
        cbar_label: The label for the colorbar
        xticks: The ticks for the horizontal axis
        yticks: The ticks for the vertical axis
        log: If True, set a logarithmic scale for the colormap
        xticklabels: The labels for the xticks
        yticklabels: The labels for the yticks
        mask_color: The color for masked data values, if any
        cbar_ticks: The tickmarks for the colorbar
        font_size: Font size is set globally with this parameter.
    """

    from matplotlib import colors, cm

    if cmap is None:
        cmap = cm.viridis

    class MidpointNormalize(colors.Normalize):

        """
        A short class which is used by image_plot to keep zero at the color-center
        of diverging colormaps.
        """

        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            """
            A short init line using inheritance
            """
            self.midpoint = midpoint
            colors.Normalize.__init__(self, vmin, vmax, clip)

        def __call__(self, value, clip=None):
            """
            Colormapping function
            """
            # ignoring masked values and all kinds of edge cases
            result, is_scalar = self.process_value(value)
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.array(np.interp(value, x, y), mask=result.mask, copy=False)

    if midpoint:
        cax = ax.imshow(data, cmap=cmap, aspect=aspect, interpolation='none',
                        norm=MidpointNormalize(midpoint=0, vmin=vmin, vmax=vmax))
    elif log:
        cax = ax.imshow(data, cmap=cmap, norm=colors.LogNorm(), aspect=aspect,
                        vmin=vmin, vmax=vmax, interpolation='none')
    else:
        cax = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect,
                        interpolation='none')

    cmap.set_bad(color=mask_color)
    cbar = fig.colorbar(cax, ax=ax, ticks=cbar_ticks)
    cbar.set_label(cbar_label, fontsize=font_size)

    ax.set_title(title, fontsize=font_size)
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)

    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    cbar.ax.tick_params(labelsize=font_size)
    ax.tick_params(labelsize=font_size)


def hist_plot(fig, ax, data, bins='auto', yscale='log', xscale='linear',
              label='', legend=True, title='', xlabel='', ylim=None,
              density=False, model_func=None, error_sig=0, model_label='',
              color='blue', model_color='orange', alpha=0.5,
              font_size='medium', **model_kwargs):

    """
    A function that calculates and plots histograms. Can also make models using
    the model_func argument, including error shading.

    Args:
        fig: The figure object to modify
        ax: The axis object to modify
        data: The data to be histogrammed
        bins: The bins for the histogram. See numpy.histogram documentation
        yscale ('log' or 'linear'): The scale for the vertical axis. Default is 'log'
        xscale ('log' or 'linear'): The scale for the horizontal axis. Default is 'linear'
        label: The legend label for the data
        legend: If True, show the legend
        title: The title for the plot
        xlabel: The label for the horizontal axis
        ylim: The limits for the vertical axis
        density: If True, report probability density instead of counts
        model_func: Default is None. Set to a callable function of the bin edges to make a model histogram.
        error_sig: Default is 0. If greater than zero, draw error shades out to error_sig sigma (significance).
        model_label: The legend label for the model
        color: The color of the drawn histogram
        model_color: The color of the drawn model (and errors)
        alpha: The transparency parameter for the error shades.
        model_kwargs: Additional kwargs not included in the above list will be passed to model_func
        font_size: Fontsize is set globally here
    """

    counts, bins = np.histogram(data, bins=bins, density=density)
    counts = np.append(counts, 0)
    ax.plot(bins, counts, label=label, drawstyle='steps-post', color=color)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_title(title, fontsize=font_size)
    ax.set_xlabel(xlabel, fontsize=font_size)
    if density:
        ax.set_ylabel('Probability Density', fontsize=font_size)
    else:
        ax.set_ylabel('Counts', fontsize=font_size)

    if model_func is not None:
        model_prob = model_func(bins, **model_kwargs)
        if density:
            model_y = model_prob / np.diff(bins)
        else:
            model_y = model_prob * np.sum(counts)
        model_y = np.append(model_y, 0)
        if error_sig:
            N = np.prod(data.shape)
            yerr = np.sqrt(N * model_prob * (1 - model_prob))
            if density:
                yerr /= (N * np.diff(bins))
            yerr = error_sig * np.append(yerr, 0)
            ax.fill_between(bins, model_y - yerr, model_y + yerr, alpha=alpha,
                            step='post', color=model_color)
            model_label += ' and %s$\sigma$ Uncertainty' % error_sig
        ax.plot(bins, model_y, label=model_label, drawstyle='steps-post', color=model_color)

    if legend:
        ax.legend(fontsize=font_size)
    if ylim is None:
        ax.set_ylim([0.9 * np.amin(counts[counts > 0]), 10 * np.amax(counts)])
    else:
        ax.set_ylim(ylim)
    ax.tick_params(labelsize=font_size)
