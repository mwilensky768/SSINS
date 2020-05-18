#! /usr/bin/env python

from SSINS import SS, INS, version, MF
from SSINS.data import DATA_PATH
from functools import reduce
import numpy as np
import argparse
from pyuvdata import UVData, UVFlag
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", nargs='*',
                    help="The visibility file(s) to process")
parser.add_argument("-s", "--streak_sig", type=float,
                    help="The desired streak significance threshold")
parser.add_argument("-o", "--other_sig", type=float,
                    help="The desired significance threshold for other shapes")
parser.add_argument("-p", "--prefix",
                    help="The prefix for output files")
parser.add_argument("-N", "--N_samp_thresh", type=int,
                    help="The N_samp_thresh parameter for the match filter")
parser.add_argument("-c", "--clobber", action='store_true',
                    help="Whether to overwrite files that have already been written")
parser.add_argument("-x", "--no_diff", action='store_false',
                    help="Flag to turn off differencing. Use if files are already time-differenced.")
args = parser.parse_args()

version_info_list = ['%s: %s, ' % (key, version.version_info[key]) for key in version.version_info]
version_hist_substr = reduce(lambda x, y: x + y, version_info_list)

# Make the SS object
ss = SS()
ss.read(args.filename, ant_str='cross', diff=args.no_diff)

# Make the INS object
ins = INS(ss)

# Clear some memory?? and make the uvflag object for storing flags later
del ss
uvd = UVData()
uvd.read(args.filename, read_data=False)
uvf = UVFlag(uvd, waterfall=True, mode='flag')
del uvd

# Write the raw data and z-scores to h5 format
ins.write(args.prefix, sep='.', clobber=args.clobber)
ins.write(args.prefix, output_type='z_score', sep='.', clobber=args.clobber)

# Flag FM radio
where_FM = np.where(np.logical_and(ins.freq_array > 87.5e6, ins.freq_array < 108e6))
ins.metric_array[:, where_FM] = np.ma.masked
ins.metric_ms = ins.mean_subtract()
ins.history += "Manually flagged the FM band. "

# Make a filter with specified settings
with open(f"{DATA_PATH}/HERA_shape_dict.yml", 'r') as shape_file:
    shape_dict = yaml.safe_load(shape_file)

sig_thresh = {shape: args.other_sig for shape in shape_dict}
sig_thresh['narrow'] = args.other_sig
sig_thresh['streak'] = args.streak_sig
mf = MF(ins.freq_array, sig_thresh, shape_dict=shape_dict, N_samp_thresh=args.N_samp_thresh)

# Do flagging
mf.apply_match_test(ins, apply_samp_thresh=True)
ins.history += "Flagged using apply_match_test on SSINS %s." % version_hist_substr

# Write outputs
ins.write(args.prefix, output_type='mask', sep='.', clobber=args.clobber)
uvf.history += ins.history
# "flags" are not helpful if no differencing was done
if args.no_diff:
    ins.write(args.prefix, output_type='flags', sep='.', uvf=uvf, clobber=args.clobber)
ins.write(args.prefix, output_type='match_events', sep='.', clobber=args.clobber)
