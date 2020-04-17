from SSINS import INS, SS, MF
from SSINS.data import DATA_PATH
from pyuvdata import UVData, UVFlag
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filelist', nargs='*',
                    help='List of gpubox, metafits, and mwaf files.')
parser.add_argument('-d', '--outdir',
                    help='The output directory of the output files, including new mwaf files if applicable')
parser.add_argument('-o', '--obsid',
                    help='The obsid of the files.')
parser.add_argument('-r', '--rfi_flag', action='store_true',
                    help='Do rfi flagging.')
parser.add_argument('-m', '--write_mwaf', action='store_true',
                    help='If RFI flagging is requested, also write out an mwaf file')
args = parser.parse_args()

ss = SS()
gpu_files = [path for path in args.filelist if ".fits" in path]
mwaf_files = [path for path in args.filelist if ".mwaf" in path]
metafits_file = [path for path in args.filelist if ".metafits" in path]
ss.read(gpu_files + metafits_file, correct_cable_len=True,
        phase_to_pointing_center=True, ant_str='cross', diff=True,
        flag_choice='original', flag_init=True)

ins = INS(ss)
ins.history += "Read in vis data: applied cable corrections and phased to pointing center. "

prefix = '%s/%s' % (args.outdir, args.obsid)
ins.write(prefix, clobber=True)

if args.rfi_flag:
    with open(f"{DATA_PATH}/MWA_EoR_Highband_shape_dict.yml", "r") as shape_file:
    	shape_dict = yaml.safe_load(shape_file)
    sig_thresh = {shape: 5 for shape in shape_dict}
    sig_thresh["narrow"] = 5
    sig_thresh["streak"] = 10
    print(f"Flagging these shapes: {shape_dict}")

    mf = MF(ins.freq_array, sig_thresh, shape_dict=shape_dict, N_samp_thresh=20)
    mf.apply_match_test(ins, apply_samp_thresh=True)

    ins.write(prefix, output_type='mask', clobber=True)
    uvd = UVData()
    uvd.read(gpu_files + metafits_file, correct_cable_len=True,
             phase_to_pointing_center=True, ant_str='cross', read_data=False)
    uvf = UVFlag(uvd, waterfall=True, mode='flag')
    ins.write(prefix, output_type='flags', uvf=uvf, clobber=True)
    ins.write(prefix, output_type='match_events', clobber=True)
    if args.write_mwaf:
        ins.write(prefix, output_type='mwaf', metafits_file=metafits_file,
                  mwaf_files=mwaf_files, Ncoarse=len(gpu_files))
