#! /usr/bin/env python

import h5py
import hdf5plugin
from SSINS import SS, INS, version, MF, util
from SSINS.data import DATA_PATH
from functools import reduce
import numpy as np
import argparse
from pyuvdata import UVData, UVFlag
import yaml
import hera_qm
from hera_qm import metrics_io

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", nargs='*',
                    help="The visibility file(s) to process")
parser.add_argument("-s", "--streak_sig", type=float,
                    help="The desired streak significance threshold")
parser.add_argument("-o", "--other_sig", type=float,
                    help="The desired significance threshold for other shapes")
parser.add_argument("-p", "--prefix",
                    help="The prefix for output files")
parser.add_argument("-t", "--tb_aggro", type=float,
                    help="The tb_aggro parameter for the match filter.")
parser.add_argument("-c", "--clobber", action='store_true',
                    help="Whether to overwrite files that have already been written")
parser.add_argument("-x", "--no_diff", default=False, action='store_true',
                    help="Flag to turn off differencing. Use if files are already time-differenced.")
parser.add_argument("-m", "--metrics_files", type=str, nargs='*', default=[],
                    help="path to file containing ant_metrics or auto_metrics readable by "
                    "hera_qm.metrics_io.load_metric_file. ex_ants here are combined "
                    "with antennas excluded via ex_ants. Flags of visibilities formed "
                    "with these antennas will be set to True.")
parser.add_argument("-y", "--a_priori_flag_yaml", default=None,
                    help="yaml file with apriori flags")
parser.add_argument("-N", "--num_baselines", type=int, default=0,
                    help="The number of baselines to read in at a time")
parser.add_argument('--ex_ants', default='', type=str,
                    help='Comma-separated list of antennas to exclude. Flags of visibilities '
                    'formed with these antennas will be set to True.')
args = parser.parse_args()

if args.metrics_files != []:
    xants = metrics_io.process_ex_ants(ex_ants=args.ex_ants,metrics_files=args.metrics_files)
    if args.a_priori_flag_yaml is not None:
        xants = list(set(list(xants) + metrics_io.read_a_priori_ant_flags(args.a_priori_flag_yaml, ant_indices_only=True)))
else:
    xants=[]

# Make the uvflag object for storing flags later, and grab bls for partial I/O
uvd = UVData()
uvd.read(args.filename, read_data=False)
if xants==None or xants==[]:
    use_ants = uvd.get_ants()
else:
    use_ants = [ant for ant in uvd.get_ants() if ant not in xants]
uvd.select(antenna_nums=use_ants)
bls = uvd.get_antpairs()
uvf = UVFlag(uvd, waterfall=True, mode='flag')
del uvd

# Make the SS object
ss = SS()
if args.num_baselines > 0:
    ss.read(args.filename, bls=bls[:args.num_baselines],
            diff=(not args.no_diff))
    ins = INS(ss)
    Nbls = len(bls)
    for slice_ind in range(args.num_baselines, Nbls, args.num_baselines):
        ss = SS()
        ss.read(args.filename, bls=bls[slice_ind:slice_ind + args.num_baselines],
                diff=(not args.no_diff))
        new_ins = INS(ss)
        ins = util.combine_ins(ins, new_ins)
else:
    ss.read(args.filename, antenna_nums=use_ants, diff=(not args.no_diff))
    ss.select(ant_str='cross')
    ins = INS(ss)

# Clear some memory??
del ss

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
mf = MF(ins.freq_array, sig_thresh, shape_dict=shape_dict, tb_aggro=args.tb_aggro)

# Do flagging
mf.apply_match_test(ins, time_broadcast=True)
ins.history += f"Flagged using apply_match_test on SSINS {version.version}."

# Write outputs
ins.write(args.prefix, output_type='mask', sep='.', clobber=args.clobber)
uvf.history += ins.history
# "flags" are not helpful if no differencing was done
if args.no_diff:
    ins.write(args.prefix, output_type='flags', sep='.', uvf=uvf, clobber=args.clobber)
ins.write(args.prefix, output_type='match_events', sep='.', clobber=args.clobber)
