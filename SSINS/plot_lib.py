"""
A simple plotting library used by SSINS.Catalog_Plot. Direct interaction with
these functions is unnecessary.
"""
import numpy as np
from astropy.time import Time
from astropy import units
from astropy.coordinates import Longitude
import warnings


def image_plot(fig, ax, data, cmap=None, vmin=None, vmax=None, title='',
               xlabel='', ylabel='', midpoint=False, aspect='auto',
               cbar_label=None, xticks=None, yticks=None, log=False,
               xticklabels=None, yticklabels=None, mask_color='white',
               cbar_ticks=None, font_size='medium', symlog=False, linthresh=1,
               extent=None, extent_time_format='jd', convert_times=True,
               lst_prec=2, extend='neither', alpha=None):

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
        symlog: If True, set a symmetric logarithmic scale for the colormap
        linthresh: region for symlog (if enabled) to act linearly to avoid divergence at zero
        xticklabels: The labels for the xticks
        yticklabels: The labels for the yticks
        mask_color: The color for masked data values, if any
        cbar_ticks: The tickmarks for the colorbar
        font_size: Font size is set globally with this parameter.
        extent: Passes to imshow to determine ticks.
        extent_time_format: a string specifying the format of the times passed in
            the extent keyword.
        convert_times: Will convert a JD to UTC or LST in radians to an LST in
            hourangle, both using astropy.
        lst_prec: Number of sig figs to keep in LST hourangle ticklabel
        extend: Whether to extend the colorbar.
        alpha: Set a transparency factor

    Note for arguments midpoint, log, symlog, linthresh:
        * Only one of these arguments can be expressed in the plot (can't have a plot with multiple different colorbar metrics).
        * If multiple of these arguments are passed, default is linear (midpoint) followed by log, symlog.
        * Linthresh only applies when using symmetrical log (symlog) and will be ignored otherwise.
    """

    from matplotlib import colors, cm

    if cmap is None:
        cmap = cm.plasma

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

    # Make sure it does the yticks correctly
    if extent is not None:
        if (extent_time_format.lower() == 'lst') and (extent[-2] < extent[-1]):
            warnings.warn("LSTs appear to cross 24 hrs. Unwrapping. If this is an error, check extent keyword in plot_lib call.")
            extent[-1] = extent[-1] - 2 * np.pi

    # colorization methods: linear, normalized log, symmetrical log
    if midpoint:
        cax = ax.imshow(data, cmap=cmap, aspect=aspect, interpolation='none',
                        norm=MidpointNormalize(midpoint=0, vmin=vmin, vmax=vmax),
                        extent=extent, alpha=alpha)
    elif log:
        cax = ax.imshow(data, cmap=cmap, norm=colors.LogNorm(), aspect=aspect,
                        vmin=vmin, vmax=vmax, interpolation='none',
                        extent=extent, alpha=alpha)
    elif symlog:
        cax = ax.imshow(data, cmap=cmap, norm=colors.SymLogNorm(linthresh), aspect=aspect,
                        vmin=vmin, vmax=vmax, interpolation='none',
                        extent=extent, alpha=alpha)
    else:
        cax = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect,
                        interpolation='none', extent=extent, alpha=alpha)

    cmap.set_bad(color=mask_color)
    cbar = fig.colorbar(cax, ax=ax, ticks=cbar_ticks, extend=extend)
    cbar.set_label(cbar_label, fontsize=font_size)

    ax.set_title(title, fontsize=font_size)
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)

    if extent is None:
        set_ticks_labels(ax, xticks, yticks, xticklabels, yticklabels)
    elif (xticks is not None) or (yticks is not None):
        warnings.warn("Plotting keyword 'extent' has been set alongside xticks "
                      "or yticks keyword. Using manual settings.")
        set_ticks_labels(ax, xticks, yticks, xticklabels, yticklabels)
    elif convert_times:
        # This case is for when extent is set, manual settings have not been made, and conversion is desired.
        # Otherwise just use what came from extent
        if extent_time_format.lower() == 'jd':
            ax.set_yticklabels([Time(ytick, format='jd').iso[:-4] for ytick in ax.get_yticks()])
        elif extent_time_format.lower() == 'lst':
            ax.set_yticklabels([Longitude(ytick * units.radian).hourangle for ytick in ax.get_yticks()])

    cbar.ax.tick_params(labelsize=font_size)
    ax.tick_params(labelsize=font_size)


def set_ticks_labels(ax, xticks, yticks, xticklabels, yticklabels):
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)


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
