from __future__ import absolute_import, division, print_function

from matplotlib import cm, use
use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


class MidpointNormalize(colors.Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        result, is_scalar = self.process_value(value)
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.array(np.interp(value, x, y), mask=result.mask, copy=False)


def error_plot(fig, ax, x, y, xerr=None, yerr=None, title='', xlabel='',
               ylabel='Counts', legend=True, label='', drawstyle='steps-mid'):

    ax.errorbar(x, y, xerr=xerr, yerr=yerr, label=label, drawstyle=drawstyle)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if legend:
        ax.legend()


def image_plot(fig, ax, data, cmap=cm.viridis, vmin=None, vmax=None, title='',
               xlabel='Frequency (Mhz)', ylabel='Time Pair',
               cbar_label=None, xticks=None, yticks=None,
               xticklabels=None, yticklabels=None, zero_mask=False,
               mask_color='white', freq_array=None, aspect=None, grid=None):

    if zero_mask:
        data = np.ma.masked_equal(data, 0)

    if vmin is None:
        vmin = np.amin(data)
    if vmax is None:
        vmax = np.amax(data)

    if cmap is cm.coolwarm:
        cax = ax.imshow(data, cmap=cmap, clim=(vmin, vmax),
                        norm=MidpointNormalize(midpoint=0, vmin=vmin, vmax=vmax))
    else:
        cax = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)

    cmap.set_bad(color=mask_color)
    cbar = fig.colorbar(cax, ax=ax)
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
    elif xlabel == 'Frequency (Mhz)':
        xticklabels = ['%.2f' % (freq_array[tick] * 10 ** (-6)) for tick in ax.get_xticks()[1:-1].astype(int)]
        xticklabels.insert(0, '0')
        xticklabels.append('0')
        ax.set_xticklabels(xticklabels)
    elif xlabel == '$\lambda u$':
        xticklabels = ['%.0f' % (grid[tick]) for tick in ax.get_xticks()[1:-1].astype(int)]
        xticklabels.insert(0, '0')
        xticklabels.append('0')
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    elif ylabel == '$\lambda v$':
        yticklabels = ['%.0f' % (grid[tick]) for tick in ax.get_yticks()[1:-1]]
        yticklabels.insert(0, '0')
        yticklabels.append('0')
        ax.set_yticklabels(yticklabels)
    if aspect is not None:
        ax.set_aspect(aspect)
    else:
        ax.set_aspect(data.shape[1] / (data.shape[0] * 5))


def scatter_plot_2d(fig, ax, x, y, title='', xlabel='', ylabel='', c=None,
                    ylim=None, cmap=None, vmin=None, vmax=None, norm=None,
                    cbar_label=None, s=None, xticks=None, yticks=None,
                    edgecolors='face'):

    cax = ax.scatter(x, y, c=c, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm, s=s,
                     edgecolors=edgecolors)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if cmap is not None:
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label(cbar_label)
    if ylim:
        ax.set_ylim(ylim)
