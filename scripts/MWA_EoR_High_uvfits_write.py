from SSINS import INS, SS, Catalog_Plot, MF, util
from astropy.time import Time
from matplotlib import cm
import numpy as np
import argparse
import os
from pyuvdata import UVData, utils, UVFlag


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--obsid', help='The obsid of the file in question')
parser.add_argument('-i', '--insfile', help='The path to the input file')
parser.add_argument('-m', '--maskfile', help='The path to the masks')
parser.add_argument('-d', '--outdir', help='The output directory')
parser.add_argument('-u', '--uvd', nargs='*', help='The path to the uvdata files')
parser.add_argument('-n', '--nsample_default', default=0, type=float, help='The default nsample to use when some nsample are 0.')
parser.add_argument('-f', '--rfi_flag', action='store_true', help="Whether or not to do rfi flagging with SSINS")
parser.add_argument('-c', '--correct', action='store_true', help="Whether to correct digital gains, bandpass shape, and digital nonlinearity.")
parser.add_argument('-t', '--time_avg', default=0, type=int, help="Number of times to average together after flagging.")
parser.add_argument('-a', '--freq_avg', default=0, type=int, help="Number of frequency channels to average together after flagging.")
args = parser.parse_args()

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

print(f"The filelist is {args.uvd}")
indir = args.uvd[0][:args.uvd[0].rfind('/')]
if indir == args.outdir:
    raise ValueError("indir and outdir are the same")

if args.rfi_flag:
    if args.insfile is not None:
        ins = INS(args.insfile, mask_file=args.maskfile)
    else:
        ss = SS()
        if args.correct:
            ss.read(args.uvd, phase_to_pointing_center=True,
                    correct_cable_len=True, flag_choice='original', diff=True,
                    remove_dig_gains=True, remove_coarse_band=True)
        else:
            ss.read(args.uvd, phase_to_pointing_center=True,
                    correct_cable_len=True, flag_choice='original', diff=True,
                    remove_dig_gains=False, remove_coarse_band=False)

        ins = INS(ss)
        prefix = f'{args.outdir}/{args.obsid}'
        ins.write(prefix)
        Catalog_Plot.INS_plot(ins, prefix, data_cmap=cm.plasma, ms_vmin=-5, ms_vmax=5,
                              title=args.obsid, xlabel='Frequency (Mhz)',
                              ylabel='Time (UTC)')

        # Try to save memory - hope for garbage collector
        del ss

        # Set up MF flagging for routine shapes
        shape_dict = {'TV6': [1.74e8, 1.81e8], 'TV7': [1.81e8, 1.88e8],
                      'TV8': [1.88e8, 1.95e8], 'TV9': [1.95e8, 2.02e8]}
        sig_thresh = {shape: 5 for shape in shape_dict}
        sig_thresh['narrow'] = 5
        sig_thresh['streak'] = 10
        mf = MF(ins.freq_array, sig_thresh, shape_dict=shape_dict,
                tb_aggro=0.5, broadcast_streak=True)

        # Do the flagging
        mf.apply_match_test(ins, event_record=True, time_broadcast=True,
                            freq_broadcast=True)
        ins.write(prefix, output_type='mask')
        Catalog_Plot.INS_plot(ins, f'{prefix}_flagged', data_cmap=cm.plasma,
                              ms_vmin=-5, ms_vmax=5, title=args.obsid,
                              xlabel='Frequency (Mhz)', ylabel='Time (UTC)')

    uvd = UVData()
    if args.correct:
        uvd.read(args.uvd, phase_to_pointing_center=True, correct_cable_len=True,
                 remove_dig_gains=True, remove_coarse_band=True, correct_van_vleck=True)
    else:
        uvd.read(args.uvd, phase_to_pointing_center=True, correct_cable_len=True,
                 remove_dig_gains=False, remove_coarse_band=False, correct_van_vleck=False)
    uvf = UVFlag(uvd, mode='flag', waterfall=True)
    uvf.flag_array = ins.mask_to_flags()
    utils.apply_uvflag(uvd, uvf, inplace=True)
    if args.time_avg > 0:
        uvd.downsample_in_time(n_times_to_avg=args.time_avg)
    if args.freq_avg > 0:
        uvd.frequency_average(args.freq_avg)

if np.any(uvd.nsample_array == 0) and (args.nsample_default > 0):
    uvd.nsample_array[uvd.nsample_array == 0] = args.nsample_default

uvd.write_uvfits(f'{args.outdir}/{args.obsid}.uvfits', spoof_nonessential=True)
