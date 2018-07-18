from matplotlib import cm, use
use('Agg')
import matplotlib.pyplot as plt
import plot_lib as pl
import Util


def INS_plot(INS, xticks=None, yticks=None, vmin=None, vmax=None,
             xticklabels=None, yticklabels=None, zero_mask=False, aspect=None):
    """
    Takes a noise spectrum and plots its relevant data products.
    Image Plots: Data/Mean-Subtracted Data
    Histograms: Mean-Subtracted Data/Events
    """

    im_kwargs = {'xticks': xticks,
                 'yticks': yticks,
                 'vmin': vmin,
                 'vmax': vmax,
                 'xticklabels': xticklabels,
                 'yticklabels': yticklabels,
                 'zero_mask': zero_mask,
                 'aspect': aspect}

    suptitles = ['%s Incoherent Noise Spectrum' % (INS.obs),
                 '%s Mean-Subtracted Incoherent Noise Spectrum' % (INS.obs)]

    data_kwargs = [{'cbar_label': 'Amplitude (%s)' % (INS.vis_units),
                    'mask_color': 'white'},
                   {'cbar_label': 'Deviation ($\hat{\sigma}$)',
                    'mask_color': 'black',
                    'cmap': cm.coolwarm}]

    for i, string in enumerate(['', '_ms']):
        for spw in range(INS.data.shape[1]):
            fig, ax = plt.subplots(figsize=(14, 8), nrows=INS.data.shape[3])
            fig.suptitle('%s, spw%i' % (suptitles[i], spw))
            im_kwargs.update(data_kwargs[i])
            for pol in range(4):
                pl.image_plot(fig, ax[pol],
                              getattr(INS, 'data%s' % (string))[:, spw, :, pol],
                              title=INS.pols[pol], **im_kwargs)
            fig.savefig('%s/figs/%s_%s_INS%s.png' % (INS.outpath, INS.obs, string))
            plt.close(fig)

    for i, string in enumerate(['match_', 'chisq_']):
        if len(getattr(INS, '%shists' % (string))):
            for k, hist in enumerate(getattr(INS, '%shists')):
                fig, ax = plt.subplots(figsize=(14, 8))
                exp, var = Util.hist_fit(hist[0], hist[1])
                x = hist[1][:-1] + 0.5 * np.diff(hist[1])
                pl.error_plot(fig, ax, x, hist[0], xlabel='Deviation ($\hat{\sigma}$)')
                pl.error_plot(fig, ax, x, exp, yerr=np.sqrt(var), xlabel='Deviation ($\hat{\sigma}$)')
                fig.savefig('%s/figs/%s_spw%i_f%i_f%i_%sevent_hist_%i.png' %
                            (outpath, INS.obs, event[0],
                             event[1].indices(INS.shape[2])[0],
                             event[1].indices(INS.shape[2])[1], string, k))
