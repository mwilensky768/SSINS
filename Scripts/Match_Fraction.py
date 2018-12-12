from __future__ import division, absolute_import, print_function

import numpy as np
from SSINS import util, plot_lib
import glob
from matplotlib import cm, colors
import matplotlib.pyplot as plt
import argparse
import pickle
import os

"""
Read in some pre-filtered noise spectra and make a scatter plot of their RFI
occupancies as a function of LST
"""

parser = argparse.ArgumentParser()
parser.add_argument('occ_dict', help='A pickled python dictionary of flagging occupancies')
parser.add_argument('times_dir', help='The directory where the LST and JD are kept')
parser.add_argument('shape', help='The shape to make the plots for')
parser.add_argument('sig_thresh', type=float, help='The significance threshold to plot for')
parser.add_argument('outdir', help='The output directory')
parser.add_argument('-g', help='A boolean array of frequency channels to keep')
parser.add_argument('--vmax', type=float, help='Clip the colorbar')
args = parser.parse_args()

if args.g is not None:
    good_freqs = np.load(args.g)

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

occ_dict = pickle.load(open(args.occ_dict, 'rb'))[args.sig_thresh][args.shape]
print(len(occ_dict.keys()))

scat_dat = np.zeros((len(occ_dict.keys()), 3))
missing_obs = []

for i, obs in enumerate(occ_dict):
    x_path = '%s/%s_lst_arr.npy' % (args.times_dir, obs)
    y_path = '%s/%s_times_arr.npy' % (args.times_dir, obs)
    if os.path.exists(x_path) and os.path.exists(y_path):
        x = np.load(x_path)[0]
        # Go from [-pi, pi]
        if x > np.pi:
            x -= 2 * np.pi
        # Convert to radians
        x *= 23.9345 / (2 * np.pi)
        y = np.load(y_path)[0]
        if args.shape == 'total':
            if args.g is not None:
                occ = np.mean(occ_dict[obs][good_freqs])
            else:
                occ = np.mean(occ_dict[obs])
        else:
            occ = occ_dict[obs]
        scat_dat[i] = [x, y, occ]
    else:
        missing_obs.append(obs)

util.make_obsfile(missing_obs, '%s/missing_obs.txt' % args.outdir)
scat_dat = scat_dat[scat_dat[:, 1] > 0]
print(scat_dat.shape)

counts, bins = np.histogram(scat_dat[:, 2], bins='auto')
counts = np.append(counts, 0)
where_0 = scat_dat[:, 2] == 0
Nz = np.count_nonzero(where_0)

fig_scat, ax_scat = plt.subplots(figsize=(8, 4.5))
fig_hist, ax_hist = plt.subplots(figsize=(8, 4.5))

if args.vmax:
    vmax = args.vmax
else:
    vmax = 0.4

thresh_obs = [key for key in occ_dict.keys() if occ_dict[key] > vmax]
util.make_obsfile(thresh_obs, '%s/thresh_obs_%.2f.txt' % (args.outdir, vmax))

plot_lib.scatter_plot_2d(fig_scat, ax_scat, scat_dat[:, 0][where_0],
                         scat_dat[:, 1][where_0], c='white',
                         edgecolors='brown', s=20)

plot_lib.scatter_plot_2d(fig_scat, ax_scat,
                         scat_dat[:, 0][np.logical_not(where_0)],
                         scat_dat[:, 1][np.logical_not(where_0)],
                         c=scat_dat[:, 2][np.logical_not(where_0)],
                         cmap=cm.copper_r,
                         title='Occupation (%s) Scatter, %i$\sigma$' % (args.shape, args.sig_thresh),
                         xlabel='LST (hours)', ylabel='Julian Date (Days)', s=20,
                         vmax=vmax, cbar_label='Occupation Fraction')

plot_lib.error_plot(fig_hist, ax_hist, bins, counts, drawstyle='steps-post',
                    title='Occupation (%s) Histogram %i$\sigma$' % (args.shape, args.sig_thresh),
                    xlabel='Occupation Fraction', ylabel='Counts',
                    label='$N_z = %i$' % Nz, legend=True, leg_size='xx-large',
                    ylim=[0.5, 2 * np.amax(counts)], yscale='log')

np.save('%s/scat_dat_%i_%s.npy' % (args.outdir, args.sig_thresh, args.shape), scat_dat)
np.save('%s/counts_%i_%s.npy' % (args.outdir, args.sig_thresh, args.shape), counts[:-1])
np.save('%s/bins_%i_%s.npy' % (args.outdir, args.sig_thresh, args.shape), bins)

fig_scat.savefig('%s/scat_%i_%s.png' % (args.outdir, args.sig_thresh, args.shape))
fig_hist.savefig('%s/hist_%i_%s.png' % (args.outdir, args.sig_thresh, args.shape))
