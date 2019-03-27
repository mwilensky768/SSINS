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
               cbar_ticks=None):

    """
    Plots 2-d images. The colormap cm.coolwarm invokes the MidpointNormalize()
    class.
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
    cbar.set_label(cbar_label)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)


def hist_plot(fig, ax, data, bins='auto', yscale='log', xscale='linear',
              label='', model_func=None, legend=True, title='', density=False,
              xlabel='', ylim=None, error_sig=0, alpha=0.5, model_label='',
              **model_kwargs):

    counts, bins, _ = ax.hist(data, bins=bins, histtype='step', label=label,
                              density=density)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if density:
        ax.set_ylabel('Probability Density')
    else:
        ax.set_ylabel('Counts')

    if model_func is not None:
        model_prob = model_func(bins, **model_kwargs)
        if density:
            model_y = model_prob / np.diff(bins)
        else:
            model_y = model_prob * np.sum(counts)
        model_y = np.append(model_y, 0)
        ax.plot(bins, model_y, label=model_label, drawstyle='steps-post')
        if error_sig:
            N = np.prod(data.shape)
            yerr = np.sqrt(N * model_prob * (1 - model_prob))
            if density:
                yerr /= (N * np.diff(bins))
            yerr = error_sig * np.append(yerr, 0)
            ax.fill_between(bins, model_y - yerr, model_y + yerr, alpha=alpha,
                            label='%s Error' % model_label, step='post')

    if legend:
        ax.legend()
    if ylim is None:
        ax.set_ylim([0.9 * np.amin(counts[counts > 0]), 10 * np.amax(counts)])
    else:
        ax.set_ylim(ylim)
