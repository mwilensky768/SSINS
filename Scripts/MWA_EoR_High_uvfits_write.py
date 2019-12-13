from SSINS import INS
import numpy as np
import argparse
import os
from pyuvdata import UVData, utils

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--obsid', help='The obsid of the file in question')
parser.add_argument('-i', '--insfile', help='The path to the input file')
parser.add_argument('-m', '--maskfile', help='The path to the masks')
parser.add_argument('-d', '--outdir', help='The output directory')
parser.add_argument('-u', '--uvd', nargs='*', help='The path to the uvdata files')
parser.add_argument('-n', '--nsample_default', default=1, type=float, help='The default nsample to use.')
args = parser.parse_args()

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

indir = args.infile[:args.infile.rfind('/')]
if indir == args.outdir:
    raise ValueError("indir and outdir are the same")

uvd = UVData()
if len(args.uvd) > 1:
    uvd.read_mwa_corr_fits(args.uvd)
else:
    uvd.read(args.uvd)
uvd.select(times=np.unique(uv.time_array)[3:-3])

ins = INS(args.insfile, mask_file=args.maskfile)
uvf = ins.copy()
uvf.to_flag()
uvf.flag_array = ins.mask_to_flags()

utils.apply_uvflag(uvd, uvf, inplace=False)
if np.any(uvd.nsample_array == 0):
    uvd.nsample_array[uvd.nsample_array == 0] = args.nsample_default

uvd.write_uvfits('%s/%s.uvfits' % (args.outdir, args.obsid))
