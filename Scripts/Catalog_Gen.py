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
ss.read(inpath, read_data=False)
times = np.unique(ss.time_array)[1:-3]
ss.read(inpath, times=times, ant_str='cross')

ins = INS(ss)


shape_dict = {'TV6': [1.74e8, 1.81e8],
              'TV7': [1.81e8, 1.88e8],
              'TV8': [1.88e8, 1.96e8],
              'broad6': [1.72e8, 1.83e8],
              'broad7': [1.79e8, 1.9e9],
              'broad8': [1.86e8, 1.97e8]}

mf = MF(ins.freq_array, 5, shape_dict=shape_dict, N_samp_thresh=15)
mf.apply_match_test(ins, apply_samp_thresh=True)

prefix = '%s/%s' % (args.outpath, args.obs)
ins.write(prefix)
ins.write(prefix, output_type='mask')
ins.write(prefix, output_type='match_events')
