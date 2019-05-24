"""
A library of wrappers which call plotting functions in the SSINS.plot_lib
library. Each function is named according to the class that they plot.
"""

from __future__ import absolute_import, division, print_function

import os
import numpy as np
from SSINS import util
from SSINS.plot_lib import image_plot, hist_plot
import platform


pol_dict_keys = np.arange(-8, 5)
pol_dict_keys = np.delete(pol_dict_keys, 8)
pol_dict_values = ['YX', 'XY', 'YY', 'XX', 'LR', 'RL', 'LL', 'RR', 'pI', 'pQ', 'pU', 'pV']
pol_dict = dict(zip(pol_dict_keys, pol_dict_values))


def INS_plot(INS, prefix, file_ext='pdf', xticks=None, yticks=None, vmin=None,
             vmax=None, ms_vmin=None, ms_vmax=None, data_cmap=None,
             xticklabels=None, yticklabels=None, aspect='auto',
             cbar_ticks=None, ms_cbar_ticks=None, cbar_label='',
             xlabel='', ylabel='', log=False):

    """Plots an incoherent noise specturm and its mean-subtracted spectrum

    Args:
        INS (INS): The INS whose data is to be plotted. *Required*
        prefix (str): A prefix for the output filepath e.g. "/outdir/plots/obsid" *Required*
        file_ext (str): The type of image file to output
        xticks (sequence): The frequency channel indices to tick in INS waterfall plots.
        yticks (sequence): The time indices to tick in INS waterfall plots.
        vmin (float): The minimum of the colormap for the INS (non-mean-subtracted)
        vmax (float): The maximum of the colormap for the INS (non-mean-subtracted)
        ms_vmin (float): The minimum of the colormap for the mean-subtracted INS
        ms_vmax (float): The maximum of the colormap for the mean-subtracted INS
        data_cmap (colormap): The colormap for the non-mean-subtracted data
        xticklabels (sequence of str): The labels for the frequency ticks
        yticklabels (sequence of str): The labels for the time ticks
        aspect (float or 'auto' or 'equal'): Set the aspect ratio of the waterfall plots.
    """

    from matplotlib import cm, use
    use('Agg')
    import matplotlib.pyplot as plt

    outdir = prefix[:prefix.rfind('/')]

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    im_kwargs = {'xticks': xticks,
                 'yticks': yticks,
                 'xticklabels': xticklabels,
                 'yticklabels': yticklabels,
                 'aspect': aspect}

    data_kwargs = [{'cbar_label': cbar_label,
                    'mask_color': 'white',
                    'vmin': vmin,
                    'vmax': vmax,
                    'cmap': data_cmap,
                    'cbar_ticks': cbar_ticks,
                    'log': log},
                   {'cbar_label': 'Deviation $\hat{\sigma}$',
                    'mask_color': 'black',
                    'cmap': cm.coolwarm,
                    'vmin': ms_vmin,
                    'vmax': ms_vmax,
                    'cbar_ticks': ms_cbar_ticks,
                    'midpoint': True}]

    fig, ax = plt.subplots(nrows=INS.metric_array.shape[2],
                           ncols=2, squeeze=False, figsize=(16, 9))

    for i, data in enumerate(['array', 'ms']):
        im_kwargs.update(data_kwargs[i])
        for pol_ind in range(INS.metric_array.shape[2]):
            image_plot(fig, ax[pol_ind, i],
                       getattr(INS, 'metric_%s' % data)[:, :, pol_ind],
                       title=pol_dict[INS.polarization_array[pol_ind]],
                       xlabel=xlabel, ylabel=ylabel, **im_kwargs)
    plt.tight_layout(h_pad=1, w_pad=1)
    fig.savefig('%s_SSINS.%s' % (prefix, file_ext))
    plt.close(fig)


def VDH_plot(SS, prefix, file_ext='pdf', xlabel='', xscale='linear', yscale='log',
             bins='auto', legend=True, ylim=None, density=False, pre_flag=True,
             post_flag=True, pre_model=True, post_model=True, error_sig=0,
             alpha=0.5, pre_label='', post_label='', pre_model_label='',
             post_model_label='', pre_color='black', post_color='blue',
             pre_model_color='purple', post_model_color='green', font_size='medium'):

    """Plots a histogram of the amplitudes of the visibility differences that
    result from sky subtraction.

    Args:
        SS (SS): An SS object whose data to plot *Required*
        prefix (str): A prefix for the output filepath of the plot e.g. /outdir/plots/obsid *Required*
        file_ext (str): The file extension for the plot. Determines the filetype of the image.
        xlabel (str): The label for the horizontal axis of the histogram
        xscale ('linear' or 'log'): The scale of the horizontal axis
        yscale ('linear' or 'log'): The scale of the vertical axis
        bins: See numpy.histogram() documentation
        legend (bool): Whether or not to display a legend
        ylim: Set the limits for the vertical axis
        density (bool): Report a probability density instead of counts
        pre_flag (bool): Plot the data without applying flags
        post_flag (bool): Plot the data after applying flags
        pre_model (bool): Plot a rayleigh-mixture fit made from data without applying flags
        post_model (bool): Plot a rayleigh-mixture fit made from data after applying flags
        error_sig (float): Plot error shades to specified number of sigma
        alpha (float): Specify alpha parameter for error shading
        pre_label (str): The legend label for amplitudes made from data without flags applied.
        post_label (str): The legend label for amplitudes made from data with flags applied.
        pre_model_label (str): The legend label for a model made from data without flags applied.
        post_model_label (str): The legend label for a model made from data with flags applied.
        pre_color (str): The color of the pre-flag histogram
        post_color (str): The color of the post-flag histogram
        pre_model_color (str): The color of the pre-flag model
        post_model_color (str): The color of the post-flag model
        font_size (str): The font size for all labels
    """
    from matplotlib import use
    use('Agg')
    import matplotlib.pyplot as plt

    outdir = prefix[:prefix.rfind('/')]

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fig, ax = plt.subplots()

    if pre_model or post_model:
        model_func = SS.mixture_prob
    else:
        model_func = None

    if post_flag and SS.flag_choice is not None:
        hist_plot(fig, ax, np.abs(SS.data_array[np.logical_not(SS.data_array.mask)]),
                  bins=bins, legend=legend, model_func=model_func,
                  yscale=yscale, ylim=ylim, density=density, label=post_label,
                  xlabel=xlabel, error_sig=error_sig, alpha=alpha,
                  model_label=post_model_label, color=post_color,
                  model_color=post_model_color, font_size=font_size)
    if pre_flag:
        if SS.flag_choice is not 'original':
            temp_flags = np.copy(SS.data_array.mask)
            temp_choice = '%s' % SS.flag_choice
        else:
            temp_choice = 'original'
        SS.apply_flags(flag_choice=None)
        hist_plot(fig, ax, np.abs(SS.data_array).flatten(), bins=bins,
                  legend=legend, model_func=model_func, yscale=yscale,
                  ylim=ylim, density=density, label=pre_label, alpha=alpha,
                  xlabel=xlabel, error_sig=error_sig, model_label=pre_model_label,
                  color=pre_color, model_color=pre_model_color, font_size=font_size)
        if temp_choice is 'original':
            SS.apply_flags(flag_choice='original')
        else:
            SS.apply_flags(flag_choice='custom', custom=temp_flags)
            SS.flag_choice = temp_choice

    fig.savefig('%s_VDH.%s' % (prefix, file_ext), bbox_inches="tight")
