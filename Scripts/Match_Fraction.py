import numpy as np
from SSINS import util
import glob
import plot_lib as pl
from matplotlib import cm
import matplotlib.pyplot as plt

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
L1 = len('%s/' % (args.meta_inpath))
L2 = len('%s/' % (args.match_events_inpath))
obslist = [path[L1:L1 + 10] for path in time_arrs]
match_event_arrs = glob.glob('%s/*match_events*' % args.match_events_inpath)
match_event_arrs = [path for path in match_event_arrs if path[L2:L2 + 10] in obslist]
for dat in [time_arrs, lst_arrs, match_event_arrs]:
    dat.sort()

JD = []
lst = []
match_event_frac = []
for time_arr, lst_arr, match_arr, freq_arr in\
        zip([time_arrs, lst_arrs, match_event_arrs]):
    JD_arr = np.load(time_arr)
    JD.append(JD_arr[0])
    lst_rad = np.load(lst_arr)[0]
    if lst_rad > np.pi:
        lst_rad -= np.pi
    lst.append(23.9345 / (2 * np.pi) * lst_rad)
    match = np.load(match_arr)
    event_frac = util.event_fraction(match, len(JD_arr))
    if slice(args.chan_min, args.chan_max) in event_frac:
        match_event_frac.append(event_frac[slice(args.chan_min, args.chan_max)])
    else:
        match_event_frac.append(0)

fig, ax = plt.subplots(figsize=(14, 8))
pl.scatter_plot_2d(fig, ax, lst, JD, c=match_event_frac, cmap=cm.viridis,
                   vmin=0, vmax=1, xlabel='LST (hours)', ylabel='JD (days)',
                   cbar_label='Occupancy Fraction')
fig.savefig(args.outpath)
