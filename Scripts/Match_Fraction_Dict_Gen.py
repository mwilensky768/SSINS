from __future__ import division, absolute_import, print_function

from SSINS import util, INS, MF
import argparse
import pickle
import glob
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('obsfile', help='A text file of obs to process')
parser.add_argument('basedir', help='The directory which hold the spectra')
parser.add_argument('outdir', help='The output directory')
parser.add_argument('flag_choice', help='The flag choice for the data to be processed')
parser.add_argument('sigs', nargs='*', type=float, help='The significance thresholds so use')
parser.add_argument('--shapes', nargs='*', help='The names of the shapes')
parser.add_argument('--mins', nargs='*', type=float, help='The minimum freqs for the shapes (in hz)')
parser.add_argument('--maxs', nargs='*', type=float, help='The maximum freqs for the shapes (in hz)')
parser.add_argument('-s', action='store_false', help='Whether _not_ to flag streaks')
parser.add_argument('-p', action='store_false', help='Whether _not_ to flag points')
parser.add_argument('-N', type=int, default=0, help='N_thres - defaults to 0')
args = parser.parse_args()

shape_dict = {}
if args.shapes is not None:
    shape_dict = {shape: [shape_min, shape_max] for (shape, shape_min, shape_max) in
                  zip(args.shapes, args.mins, args.maxs)}

obslist = util.make_obslist(args.obsfile)
if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

missing_obs = []
shape_list = args.shapes + ['total']
if args.s:
    shape_list.insert(-1, 'streak')
if args.p:
    shape_list.insert(-1, 'point')
occ_dict = {sig: {shape: {} for shape in shape_list} for sig in args.sigs}

for obs in obslist:
    pathlist = glob.glob('%s/arrs/%s*' % (args.basedir, obs))
    if len(pathlist):
        read_paths = util.read_paths_construct(args.basedir, args.flag_choice, obs, 'INS')
        for sig in args.sigs:
            ins = INS(read_paths=read_paths, obs=obs)
            mf = MF(ins, sig_thresh=sig, N_thresh=args.N, shape_dict=shape_dict,
                    point=args.p, streak=args.s)
            mf.apply_match_test(apply_N_thresh=True)
            if len(ins.match_events):
                event_frac = util.event_fraction(ins.match_events, ins.data.shape[0],
                                                 shape_list, ins.data.shape[2])
                for shape in shape_list:
                    occ_dict[sig][shape][obs] = event_frac[shape]
            else:
                for shape in shape_list:
                    occ_dict[sig][shape][obs] = 0
            occ_dict[sig]['total'][obs] = np.mean(ins.data.mask[:, 0, :, 0], axis=0)
            del ins
            del mf
    else:
        missing_obs.append(obs)

util.make_obsfile(missing_obs, '%s/missing_obs.txt' % args.outdir)
pickle.dump(occ_dict, open('%s/occ_dict.pik' % args.outdir, 'wb'))
