from SSINS import INS, SS, MF
from SSINS import Catalog_Plot as cp
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('obsid', help='The obsid of the file in question')
parser.add_argument('infile', help='The path to the input file')
parser.add_argument('outdir', help='The output directory')
args = parser.parse_args()

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

indir = args.infile[:args.infile.rfind('/')]
if indir == args.outdir:
    raise ValueError("indir and outdir are the same")

ss = SS()
ss.read(args.infile, flag_choice='original')

ins = INS(ss)

ins.metric_array[-5:] = np.ma.masked
ins.metric_ms = ins.mean_subtract()

shape_dict = {'TV4': [1.74e8, 1.81e8],
              'TV5': [1.81e8, 1.88e8],
              'TV6': [1.88e8, 1.95e8],
              'broad6': [1.72e8, 1.83e8],
              'broad7': [1.79e8, 1.9e8],
              'broad8': [1.86e8, 1.98e8]}

mf = MF(ins.freq_array, 5, shape_dict=shape_dict, N_samp_thresh=25)
mf.apply_match_test(ins, apply_samp_thresh=True)

cp.INS_plot(ins, '%s/%s' % (args.outdir, args.obsid))

ss.apply_flags(flag_choice='INS', INS=ins)
ss.write('%s/%s.uvfits' % (args.outdir, args.obsid), 'uvfits',
         nsample_default=16)
