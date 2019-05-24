import argparse
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from SSINS import INS, util
import numpy as np
import os
import copy


def rad_convert(lst, branch):
    conv = 23.9345 / (2 * np.pi)

    if branch:
        if lst > np.pi:
            lst -= 2 * np.pi

    lst *= conv

    return(lst)


def total_occ(ins, good_chans=None):

    if good_chans is not None:
        occ = np.mean(ins.metric_array.mask[:, good_chans])
    else:
        occ = np.mean(ins.metric_array.mask)

    return(occ)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--shapes', nargs='*',
                        help="The shapes to calculate occupancy for")
    parser.add_argument('-f', '--filelist',
                        help="The list of paths to the SSINS data files, in the same order as obslist")
    parser.add_argument('-r', '--rad_convert', action='store_true',
                        help="Convert lst from radians to hours")
    parser.add_argument('-b', '--branch', action='store_true',
                        help="Map lst from [-pi, pi] instead of [0, 2pi]")
    parser.add_argument('-o', '--outdir',
                        help="The output directory. Will be made if non-existant!")
    parser.add_argument('-i', '--ch_ignore',
                        help="Text file with channels to ignore")
    parser.add_argument('-e', '--red_events', nargs='*', action='append',
                        help="Redundant events for red_event_sort")
    args = parser.parse_args()

    times = []
    lsts = []
    occ_dict = {shape: [] for shape in args.shapes}

    filelist = util.make_obslist(args.filelist)
    if args.ch_ignore:
        ch_ignore = util.make_obslist(args.ch_ignore)
        ch_ignore = [int(ch) for ch in ch_ignore]
    else:
        ch_ignore = None

    if args.red_events:
        red_events = [tuple(lis) for lis in args.red_events]

    for path in filelist:
        prefix = path[:path.rfind('_')]
        match_events_file = '%s_match_events.yml' % prefix
        mask_file = '%s_mask.h5' % prefix
        ins = INS(path, mask_file=mask_file, match_events_file=match_events_file)

        times.append(ins.time_array[0])

        lst = ins.lst_array[0]
        if args.rad_convert:
            lst = rad_convert(lst, args.branch)
        elif args.branch:
            if lst > np.pi:
                lst -= 2 * np.pi
        lsts.append(lst)

        if 'total' in args.shapes:
            if ch_ignore is not None:
                good_chans = np.ones(len(ins.freq_array), dtype=bool)
                good_chans[ch_ignore] = 0
            else:
                good_chans = None
            occ_dict['total'].append(total_occ(ins, good_chans))
            event_shapes = copy.deepcopy(args.shapes)
            event_shapes.remove('total')
        else:
            event_shapes = args.shapes
        if event_shapes:
            if args.red_events:
                match_events = util.red_event_sort(ins.match_events, red_events)
            else:
                match_events = ins.match_events
            occs = util.event_fraction(ins.match_events, len(ins.time_array),
                                       event_shapes, len(ins.freq_array))
            for shape in event_shapes:
                occ_dict[shape].append(occs[shape])

    #winter = cm.get_cmap('winter', 256)
    #newcolors = winter(np.linspace(0, 1, 256))
    #black = np.array([0, 0, 0, 1])
    #newcolors[0] = black
    #cmap = ListedColormap(winter)

    cmap = cm.winter
    cmap.set_under('black')

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    for shape in args.shapes:
        fig, ax = plt.subplots()
        occ_arr = np.array(occ_dict[shape])
        lim = np.amin(occ_arr[occ_arr > 0])
        cax = ax.scatter(lsts, times, c=occ_dict[shape], cmap=cmap, s=8, vmin=lim,
                         vmax=0.4)
        ax.set_xlabel('LST (hours)')
        ax.set_ylabel('JD (days)')
        ax.set_title('Total RFI Occupancy')
        fig.colorbar(cax, ax=ax, extend='min', label='Occupancy Fraction')
        fig.savefig('%s/%s_occ_scatter.pdf' % (args.outdir, shape))
