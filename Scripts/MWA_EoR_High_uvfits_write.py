from SSINS import INS
import numpy as np
import argparse
import os
from pyuvdata import UVData, utils


def uvd_read_select(obj, filepath, time_slice=slice(3, -3)):
    obj.read(filepath, read_data=False)
    times = np.unique(obj.time_array)[time_slice]
    obj.read(filepath, times=times)


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--obsid', help='The obsid of the file in question')
parser.add_argument('-i', '--insfile', help='The path to the input file')
parser.add_argument('-m', '--maskfile', help='The path to the masks')
parser.add_argument('-d', '--outdir', help='The output directory')
parser.add_argument('-u', '--uvd', nargs='*', help='The path to the uvdata files')
parser.add_argument('-n', '--nsample_default', default=1, type=float, help='The default nsample to use.')
parser.add_argument('-f', '--rfi_flag', type='store_true', help="Whether or not to do rfi flagging with SSINS")
args = parser.parse_args()

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

indir = args.infile[:args.infile.rfind('/')]
if indir == args.outdir:
    raise ValueError("indir and outdir are the same")

if args.rfi_flag:
    if args.insfile is not None:
        ins = INS(args.insfile, mask_file=args.maskfile)
    else:
        ss = SS()
        uvd_obj_read(ss, args.uvd)
        ins = INS(ss)
        freqs = np.arange(1.7e8, 2e8, 5e6)
        xticks, xticklabels = util.make_ticks_labels(freqs, ins.freq_array, sig_fig=0)
        yticks = [0, 20, 40]
        yticklabels = []
        for tick in yticks:
            yticklabels.append(Time(ins.time_array[tick], format='jd').iso[:-4])
        Catalog_Plot.INS_plot(ins, '%s/%s' % (args.outdir, args.obsid),
                              xticks=xticks, yticks=yticks, xticklabels=xticklabels,
                              yticklabels=yticklabels, data_cmap=cm.plasma,
                              ms_vmin=-5, ms_vmax=5, title=obsid,
                              xlabel='Frequency (Mhz)', ylabel='Time (UTC)')
        # Try to save memory - hope for garbage collector
        del ss
        # Set up MF flagging for routine shapes
        shape_dict = {'TV6': [1.74e8, 1.81e8], 'TV7': [1.81e8, 1.88e8],
                      'TV8': [1.88e8, 1.95e8], 'TV9': [1.95e8, 2.02e8]}
        sig_thresh = {shape: 5 for shape in shape_dict}
        sig_thresh['narrow'] = 5
        sig_thresh['streak'] = 8
        mf = MF(ins.freq_array, sig_thresh, shape_dict=shape_dict,
                N_samp_thresh=len(ins.time_array) // 2)
        mf.apply_match_test(ins, apply_samp_thresh=False)
        mf.apply_samp_thresh_test(ins, event_record=True)
        Catalog_Plot.INS_plot(ins, '%s/%s_flagged' % (args.outdir, args.obsid),
                              xticks=xticks, yticks=yticks, xticklabels=xticklabels,
                              yticklabels=yticklabels, data_cmap=cm.plasma,
                              ms_vmin=-5, ms_vmax=5, title=obsid,
                              xlabel='Frequency (Mhz)', ylabel='Time (UTC)')

    uvd = UVData()
    uvd_obj_read(uvd, args.uvd)
    uvf = ins.copy()
    uvf.to_flag()
    uvf.flag_array = ins.mask_to_flags()
    utils.apply_uvflag(uvd, uvf, inplace=False)

if np.any(uvd.nsample_array == 0):
    uvd.nsample_array[uvd.nsample_array == 0] = args.nsample_default

uvd.write_uvfits('%s/%s.uvfits' % (args.outdir, args.obsid))
