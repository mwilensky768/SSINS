from __future__ import divide, absolute_import, print_function

import numpy as np
from SSINS import util
import glob
from SSINS import plot_lib as pl
from matplotlib import cm, colors
import matplotlib.pyplot as plt
import argparse

"""
Read in some pre-filtered noise spectra and make a scatter plot of their RFI
occupancies as a function of LST
"""

parser = argparse.ArgumentParser()
parser.add_argument('match_events_inpath', help='The input base directory')
parser.add_argument('meta_inpath', help='The base directory with the time arrays')
parser.add_argument('outpath', help='The target filename for the plot, should end in a valid matplotlib filetype')
parser.add_argument('chan_min', type=int)
parser.add_argument('chan_max', type=int)

args = parser.parse_args()


time_arrs = glob.glob('%s/*time*' % (args.meta_inpath))
lst_arrs = glob.glob('%s/*lst*' % (args.meta_inpath))
freq_arrs = glob.glob('%s/*freq_array*' % (args.meta_inpath))
L2 = len('%s/' % (args.meta_inpath))
L1 = len('%s/' % (args.match_events_inpath))
match_event_arrs = glob.glob('%s/*match_events*' % args.match_events_inpath)
obslist = [path[L1:L1 + 10] for path in match_event_arrs]
master_obslist = [path[L2:L2 + 10] for path in time_arrs]
for dat in [time_arrs, lst_arrs, match_event_arrs, freq_arrs]:
    dat.sort()

JD = []
lst = []
match_event_frac = []
max_JD_len = 0
for i, obs in enumerate(master_obslist):
    JD_arr = np.load(time_arrs[i])
    if len(JD_arr) > max_JD_len:
        max_JD_len = len(JD_arr)
    JD.append(JD_arr[0])
    lst_rad = np.load(lst_arrs[i])[0]
    if lst_rad > np.pi:
        lst_rad -= 2 * np.pi
    lst.append(23.9345 / (2 * np.pi) * lst_rad)
    Nfreqs = len(np.load(freq_arrs[i])[0])
    if obs in obslist:
        k = np.where(np.array(obslist) == obs)[0][0]
        match = np.load(match_event_arrs[k])
        event_frac = util.event_fraction(match, Nfreqs, len(JD_arr) - 1)
        if (args.chan_min, args.chan_max) in event_frac:
            match_event_frac.append(event_frac[(args.chan_min, args.chan_max)])
        else:
            match_event_frac.append(0)
    else:
        match_event_frac.append(0)

p_obs = [1061313616, 1061315448, 1061317272, 1061663760]

point_switch = []
for obs in p_obs:
    lst_p = np.load('/Users/mike_e_dubs/MWA/INS/Long_Run/All/metadata/%i_lst_array.npy' % (obs))[0]
    if lst_p > np.pi:
        lst_p -= 2 * np.pi
    lst_p *= 23.9345 / (2 * np.pi)
    point_switch.append(lst_p)

fig, ax = plt.subplots(figsize=(14, 8))
bool_0 = (np.array(match_event_frac) == 0).astype(bool)
bool_1 = np.logical_not(bool_0)
lst_0 = np.array(lst)[bool_0]
JD_0 = np.array(JD)[bool_0]
lst_1 = np.array(lst)[bool_1]
JD_1 = np.array(JD)[bool_1]
mef_1 = np.array(match_event_frac)[bool_1]
pl.scatter_plot_2d(fig, ax, lst_0, JD_0, c='white', edgecolors='brown',
                   vmin=1 / max_JD_len, vmax=0.4, xlabel='LST (hours)', ylabel='JD (days)',
                   cbar_label='Occupancy Fraction')
pl.scatter_plot_2d(fig, ax, lst_1, JD_1, c=mef_1, cmap=cm.copper_r,
                   vmin=1 / max_JD_len, vmax=0.4, xlabel='LST (hours)', ylabel='JD (days)',
                   cbar_label='Occupancy Fraction')
for switch in point_switch:
    ax.axvline(x=switch, color='black')
fig.savefig(args.outpath)
