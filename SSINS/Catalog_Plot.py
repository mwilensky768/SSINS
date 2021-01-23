"""
A library of wrappers which call plotting functions in the SSINS.plot_lib
library. Each function is named according to the class that they plot.
"""
import os
import numpy as np
from SSINS.plot_lib import image_plot, hist_plot
import warnings


pol_dict_keys = np.arange(-8, 5)
pol_dict_keys = np.delete(pol_dict_keys, 8)
pol_dict_values = ['YX', 'XY', 'YY', 'XX', 'LR', 'RL', 'LL', 'RR', 'pI', 'pQ', 'pU', 'pV']
pol_dict = dict(zip(pol_dict_keys, pol_dict_values))


def INS_plot(INS, prefix, file_ext='pdf', xticks=None, yticks=None, vmin=None,
             vmax=None, ms_vmin=None, ms_vmax=None, data_cmap=None,
             xticklabels=None, yticklabels=None, aspect='auto',
             cbar_ticks=None, ms_cbar_ticks=None, cbar_label='',
             xlabel='', ylabel='', log=False, sig_event_plot=True,
             sig_event_vmax=None, sig_event_vmin=None, sig_log=True,
             sig_cmap=None, symlog=False, linthresh=1, sample_sig_vmin=None,
             sample_sig_vmax=None, title=None, title_x=0.5, title_y=.98,
             use_extent=True, backend=None, extent_time_format='jd',
             convert_times=True):

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
        title (str): The title to use for the plot.
        title_x (float): x-coordinate of title center (in figure coordinates)
        title_y (float): y-coordinate of title center (in figure coordinates)
        use_extent (bool): Whether to use the INS metadata to make ticks on plots.
            Easier than manual adjustment and sufficient for most cases.
            Will put time in UTC and frequency in MHz.
        backend (str): Which matplotlib backend to use.
        extent_time_format (str): If 'jd', will use the time_array of the object.
            is 'lst', will use the lst_array of the object.
        convert_times (bool): Whether to convert times in extent. Will convert jd into
            UTC and lst into hourangles.
    """

    from matplotlib import cm, use
    if backend is not None:
        use(backend)
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    outdir = prefix[:prefix.rfind('/')]

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if use_extent:
        # Have to put times in reverse since vertical axis is inverted
        extent = [INS.freq_array[0] / 1e6, INS.freq_array[-1] / 1e6]
        if extent_time_format.lower() == 'jd':
            extent.extend([INS.time_array[-1], INS.time_array[0]])
        elif extent_time_format.lower() == 'lst':
            extent.extend([INS.lst_array[-1], INS.lst_array[0]])
        xlabel = "Frequency (MHz)"
        ylabel = "Time (UTC)"
    else:
        extent = None

    im_kwargs = {'xticks': xticks,
                 'yticks': yticks,
                 'xticklabels': xticklabels,
                 'yticklabels': yticklabels,
                 'extent': extent,
                 'extent_time_format': extent_time_format,
                 'convert_times': convert_times,
                 'aspect': aspect,
                 'xlabel': xlabel,
                 'ylabel': ylabel}

    data_kwargs = [{'cbar_label': cbar_label,
                    'mask_color': 'white',
                    'vmin': vmin,
                    'vmax': vmax,
                    'cmap': data_cmap,
                    'cbar_ticks': cbar_ticks,
                    'log': log,
                    'symlog': symlog,
                    'linthresh': linthresh},
                   {'cbar_label': 'Deviation ($\hat{\sigma})$',
                    'mask_color': 'black',
                    'cmap': cm.coolwarm,
                    'vmin': ms_vmin,
                    'vmax': ms_vmax,
                    'cbar_ticks': ms_cbar_ticks,
                    'midpoint': True}]

    sig_event_kwargs = [{'cbar_label': 'Significance ($\hat{\sigma}$)',
                         'vmin': sig_event_vmin,
                         'vmax': sig_event_vmax,
                         'log': sig_log,
                         'cmap': sig_cmap,
                         'midpoint': False},
                        {'cbar_label': 'Event Index',
                         'cmap': cm.viridis_r,
                         'mask_color': 'white',
                         'midpoint': False,
                         'log': False,
                         'symlog': False},
                        {'cbar_label': 'Significance ($\hat{\sigma}$)',
                         'vmin': sample_sig_vmin,
                         'vmax': sample_sig_vmax,
                         'midpoint': True,
                         'cmap': cm.coolwarm,
                         'mask_color': 'black'}]

    fig, ax = plt.subplots(nrows=INS.metric_array.shape[2],
                           ncols=2, squeeze=False, figsize=(16, 9))
    if title is not None:
        fig.suptitle(title, x=title_x, y=title_y)

    for data_ind, data in enumerate(['array', 'ms']):
        im_kwargs.update(data_kwargs[data_ind])
        for pol_ind in range(INS.metric_array.shape[2]):
            image_plot(fig, ax[pol_ind, data_ind],
                       getattr(INS, 'metric_%s' % data)[:, :, pol_ind],
                       title=pol_dict[INS.polarization_array[pol_ind]],
                       **im_kwargs)
    plt.tight_layout(h_pad=1, w_pad=1)
    fig.savefig(f'{prefix}_SSINS.{file_ext}')
    plt.close(fig)

    if sig_event_plot:
        if len(INS.match_events):
            fig, ax = plt.subplots(nrows=INS.metric_array.shape[2],
                                   ncols=3, squeeze=False, figsize=(16, 9))
            event_sig_arr = np.zeros(INS.metric_array.shape)
            event_ind_arr = np.ma.zeros(INS.metric_array.shape, dtype=int)

            # iterate backwards so that the most significant events are shown at the topmost layer
            for event_ind in range(len(INS.match_events) - 1, -1, -1):
                event_sig_arr[INS.match_events[event_ind][:2]] = INS.match_events[event_ind][-1]
                event_ind_arr[INS.match_events[event_ind][:2]] = event_ind
            event_sig_arr_wh_0 = np.where(event_sig_arr == 0)
            event_sig_arr[event_sig_arr_wh_0] = INS.metric_ms[event_sig_arr_wh_0]
            event_ind_arr[event_sig_arr_wh_0] = np.ma.masked

            for data_ind, data in enumerate([event_sig_arr, event_ind_arr, INS.sig_array]):
                im_kwargs.update(sig_event_kwargs[data_ind])
                for pol_ind in range(INS.metric_array.shape[2]):
                    image_plot(fig, ax[pol_ind, data_ind], data[:, :, pol_ind],
                               title=pol_dict[INS.polarization_array[pol_ind]],
                               **im_kwargs)
            plt.tight_layout(h_pad=1, w_pad=1)
            fig.savefig('%s_SSINS_sig.%s' % (prefix, file_ext))
            plt.close(fig)


def VDH_plot(SS, prefix, file_ext='pdf', xlabel='', xscale='linear', yscale='log',
             bins='auto', legend=True, ylim=None, density=False, pre_flag=True,
             post_flag=True, pre_model=True, post_model=True, error_sig=0,
             alpha=0.5, pre_label='', post_label='', pre_model_label='',
             post_model_label='', pre_color='orange', post_color='blue',
             pre_model_color='purple', post_model_color='green',
             font_size='medium', backend=None):

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
        backend (str): Which matplotlib backend to use.
    """
    from matplotlib import use
    if backend is not None:
        use(backend)
    import matplotlib.pyplot as plt

    outdir = prefix[:prefix.rfind('/')]

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fig, ax = plt.subplots()

    if post_flag:
        if SS.flag_choice is None:
            warnings.warn("Asking to plot post-flagging data, but SS.flag_choice is None. This is identical to plotting pre-flagging data")
        if post_model:
            model_func = SS.mixture_prob
        else:
            model_func = None
        hist_plot(fig, ax, np.abs(SS.data_array[np.logical_not(SS.data_array.mask)]),
                  bins=bins, legend=legend, model_func=model_func,
                  yscale=yscale, ylim=ylim, density=density, label=post_label,
                  xlabel=xlabel, error_sig=error_sig, alpha=alpha,
                  model_label=post_model_label, color=post_color,
                  model_color=post_model_color, font_size=font_size)
    if pre_flag:
        if pre_model:
            model_func = SS.mixture_prob
        else:
            model_func = None
        if SS.flag_choice != 'original':
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
        if temp_choice == 'original':
            SS.apply_flags(flag_choice='original')
        else:
            SS.apply_flags(flag_choice='custom', custom=temp_flags)
            SS.flag_choice = temp_choice

    fig.savefig('%s_VDH.%s' % (prefix, file_ext), bbox_inches="tight")
