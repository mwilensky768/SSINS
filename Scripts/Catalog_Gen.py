from __future__ import absolute_import, division, print_function

import argparse
from SSINS import Catalog_Plot as cp
from SSINS import SS, INS, MF
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('obs', action='store', help='The observation ID')
parser.add_argument('inpath', action='store', help='The path to the data file')
parser.add_argument('outpath', action='store', help='The base directory for saving all outputs')
args = parser.parse_args()

ss = SS()
ss.read(args.inpath, ant_str='cross')

ins = INS(ss)
ins.metric_array[:, :82] = np.ma.masked
ins.metric_array[:, :-21] = np.ma.masked
ins.order = 1
ins.metric_ms = ins.mean_subtract()

shape_dict = {'TV4': [1.74e8, 1.82e8],
              'TV5': [1.82e8, 1.9e8],
              'TV6': [1.9e8, 1.98e8],
              'dig1': [1.125e8, 1.15625e8],
              'dig2': [1.375e8, 1.40625e8],
              'dig3': [1.625e8, 1.65625e8],
              'dig4': [1.875e8, 1.90625e8]}

mf = MF(ins.freq_array, 5, shape_dict=shape_dict, N_samp_thresh=15)
mf.apply_match_test(ins, apply_samp_thresh=True)

prefix = '%s/%s' % (args.outpath, args.obs)
ins.write(prefix)
ins.write(prefix, output_type='mask')
ins.write(prefix, output_type='match_events')
