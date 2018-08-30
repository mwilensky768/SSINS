import argparse
from SSINS import INS
from SSINS import MF
from SSINS import Catalog_Plot as cp
import numpy as np
import glob

"""
A script which takes a single inpath and outpath, expecting noise spectra dispersed
in the manner that they are saved by INS.save(). This optionally applies a match
filter of specified shape/sigma_thresh and saves the outputs. It then plots them.
"""


parser = argparse.ArgumentParser()
parser.add_argument('inpath', action='store', help='The input base directory')
parser.add_argument('outpath', action='store', help='The output base directory')
parser.add_argument('Lobs', action='store', type=int, help='Length of the obsid name')
parser.add_argument('flag_choice', action='store', help='The flag choice of the observations')
parser.add_argument('--tests', action='store', nargs='*', help='Which tests to perform')
parser.add_argument('--sig_thresh', action='store', type=float,
                    help='The sigma threshold for the match_shape flagger')
parser.add_argument('--labels', action='store', nargs='*',
                    help='The labels for the shapes in the dictionary')
parser.add_argument('--mins', action='store', nargs='*', type=float,
                    help='The corresponding minimum frequencies for labels')
parser.add_argument('--maxs', action='store', nargs='*', type=float,
                    help='The corresponding maximum frequencies for labels')
parser.add_argument('--N_thresh', action='store', type=int,
                    help='The minimum number of samples of unflagged samples for a channel to be valid')
parser.add_argument('--point', action='store_true', help='Specify whether single-point outliers are to be examined')
parser.add_argument('--streak', action='store_true', help='Specify whether broadband streaks are to be examined')
args = parser.parse_args()
shape_dict = {}
if args.labels is not None:
    for label, min, max in zip(args.labels, args.mins, args.maxs):
        shape_dict[label] = np.array([min, max])

data_arrs = glob.glob('%s/arrs/*%s_INS_data.npym' % (args.inpath, args.flag_choice))
mf_kwargs = {}
for attr in ['sig_thresh', 'N_thresh']:
    if hasattr(args, attr):
        mf_kwargs[attr] = getattr(args, attr)

for arr in data_arrs:
    L = len('%s/arrs/' % (args.inpath))
    obs = arr[L:L + args.Lobs]
    read_paths = {'data': arr,
                  'Nbls': '%s/arrs/%s_%s_INS_Nbls.npym' % (args.inpath, obs, args.flag_choice),
                  'freq_array': '%s/metadata/%s_freq_array.npy' % (args.inpath, obs),
                  'pols': '%s/metadata/%s_pols.npy' % (args.inpath, obs),
                  'vis_units': '%s/metadata/%s_vis_units.npy' % (args.inpath, obs)}
    ins = INS(obs=obs, outpath=args.outpath, flag_choice=args.flag_choice, read_paths=read_paths)
    cp.INS_plot(ins)
    mf = MF(ins, shape_dict=shape_dict, point=args.point, streak=args.streak, **mf_kwargs)
    for test in args.tests:
        getattr(mf, 'apply_%s_test' % test)()
    ins.save()
    cp.MF_plot(mf)
