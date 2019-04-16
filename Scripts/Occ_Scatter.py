import argparse
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from SSINS import INS, util
import numpy as np
import os


def rad_convert(lst, branch):
    conv = 23.9345 / (2 * np.pi)

    if branch:
        if lst > np.pi:
            lst -= 2 * np.pi

    lst *= conv

    return(lst)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--shapes', nargs='*', help="The shapes to calculate occupancy for")
    parser.add_argument('-f', '--filelist', help="The list of paths to the SSINS data files, in the same order as obslist")
    parser.add_argument('-r', '--rad_convert', action='store_true', help="Convert lst from radians to hours")
    parser.add_argument('-b', '--branch', action='store_true', help="Map lst from [-pi, pi] instead of [0, 2pi]")
    parser.add_argument('-o', '--outdir', help="The output directory. Will be made if non-existant!")
    args = parser.parse_args()

    times = []
    lsts = []
    occ_dict = {shape: [] for shape in args.shapes}

    filelist = util.make_obslist(args.filelist)

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

        occs = util.event_fraction(ins.match_events, len(ins.time_array),
                                   args.shapes, len(ins.freq_array))
        for shape in args.shapes:
            occ_dict[shape].append(occs[shape])

    winter = cm.get_cmap('winter', 256)
    newcolors = winter(np.linspace(0, 1, 256))
    black = np.array([0, 0, 0, 1])
    newcolors[0] = black
    cmap = ListedColormap(newcolors)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    for shape in args.shapes:
        fig, ax = plt.subplots()
        ax.scatter(times, lsts, c=occ_dict[shape], cmap=cmap)
        fig.savefig('%s_%s_occ_scatter.pdf' % (args.outdir, shape))
