from __future__ import division, print_function, absolute_import

from SSINS import INS, util, plot_lib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from SSINS.Catalog_Plot import pol_dict


def make_sig_plots(outdir, dlist, elist, freqs, sig_fig, obslist):

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    for dpath, epath, obs in zip(dlist, elist, obslist):
        ins = INS(dpath, match_events_file=epath)
        shape_sig_arr = np.ma.copy(ins.metric_ms)
        tcolor_arr = np.ma.zeros(ins.metric_ms.shape)

        Nevent = len(ins.match_events)
        for event_ind, event in enumerate(ins.match_events):
            shape_sig_arr[event[:2]] = event[-1]
            tcolor_arr[event[:2]] = event_ind
        tcolor_arr[tcolor_arr == 0] = np.ma.masked

        fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(16, 9))
        xticks, xticklabels = util.make_ticks_labels(freqs, ins.freq_array, sig_fig=0)
        for pol_ind in range(4):
            plot_lib.image_plot(fig, ax[pol_ind][0], shape_sig_arr[:, :, pol_ind],
                                ylabel='Time (2 s)', xlabel='Frequency (Mhz)',
                                xticks=xticks, xticklabels=xticklabels,
                                cbar_label='Deviation ($\hat{\sigma}$)',
                                vmin=vmin, vmax=vmax, cmap=cm.coolwarm, symlog=True,
                                title=pol_dict[ins.polarization_array[pol_ind]])
            plot_lib.image_plot(fig, ax[pol_ind][1], tcolor_arr[:, :, pol_ind],
                                ylabel='Time (2 s)', xlabel='Frequency (Mhz)',
                                xticks=xticks, xticklabels=xticklabels,
                                cbar_label='Flagging Iteration',
                                cmap=cm.viridis_r, mask_color='white',
                                title=pol_dict[ins.polarization_array[pol_ind]])
        fig.savefig('%s/%s_flag_metaplot.pdf' % (outdir, obs))
        plt.close(fig)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dlist', required=True,
                        help="Path to a text file where each line is a path to saved SSINS data")
    parser.add_argument('-e', '--elist', required=True,
                        help="Path to a text file where each line is a path to saved match_events yaml")
    parser.add_argument('-o', '--outdir', required=True,
                        help="The output directory. Puts out plots.")
    parser.add_argument('-v', '--vmax', type=float, default=0,
                        help="The vmax for the color plots")
    parser.add_argument('-f', '--freqs', required=True,
                        help="Path to text file of frequencies to tick on plots.")
    parser.add_argument('-s', '--sig_fig', default=0,
                        help="Number of significant figures on frequency labels.")
    parser.add_argument('-l', '--obslist', required=True,
                        help="Path to a text file of obsids")
    args = parser.parse_args()

    dlist = util.make_obslist(args.dlist)
    elist = util.make_obslist(args.elist)
    obslist = util.make_obslist(args.obslist)
    freqs = np.array(util.make_obslist(args.freqs)).astype(float)
    outdir = args.outdir
    sig_fig = args.sig_fig
    if args.vmax == 0:
        vmax = None
        vmin = None
    else:
        vmax = args.vmax
        vmin = -args.vmax

    make_sig_plots(outdir, dlist, elist, freqs, sig_fig, obslist)
