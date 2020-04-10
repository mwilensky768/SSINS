from SSINS import INS, SS, MF
from SSINS.data import DATA_PATH
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
ins.write(prefix)

if args.rfi_flag:
    shape_dict = yaml.safe_load(f"{DATA_PATH}/MWA_EoR_Highband_shape_dict.yml")
    sig_thresh = {shape: 5 for shape in shape_dict}
    sig_thresh["narrow"] = 5
    sig_thresh["streak"] = 10

    mf = MF(ins.freq_array, sig_thresh, shape_dict=shape_dict, N_samp_thresh=20)
    mf.apply_match_test(ins, apply_samp_thresh=True)

    ins.write(prefix, output_type='flags')
    ins.write(prefix, output_type='match_events')
    if args.write_mwaf:
        ins.write(prefix, output_type='mwaf', metafits_file=metafits_file,
                  mwaf_files=mwaf_files, Ncoarse=len(gpu_files))
