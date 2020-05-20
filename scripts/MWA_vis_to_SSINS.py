from SSINS import INS, SS, MF
from SSINS.data import DATA_PATH
from pyuvdata import UVData, UVFlag
import yaml
import argparse
from astropy.io import fits
import numpy as np


def calc_occ(ins, num_init_flag, num_int_flag, shape_dict):

    occ_dict = {}
    # Figure out the total occupancy sans initial flags
    total_data = np.prod(ins.metric_array.shape)
    total_valid = total_data - num_init_flag
    total_flag = np.sum(ins.metric_array.mask)
    total_RFI = total_flag - num_init_flag
    total_occ = total_RFI / total_valid
    occ_dict['total'] = total_occ

    # initialize
    for shape in shape_dict:
        occ_dict[shape] = 0
    for shape in ['streak', 'narrow', 'samp_thresh']:
        occ_dict[shape] = 0

    for event in ins.match_events:
        if event[2] in ("narrow", "samp_thresh"):
            occ_dict[event[2]] += 1. / total_valid
        else:
            occ_dict[event[2]] += 1. / (ins.metric_array.shape[0] - num_int_flag)

    for item in occ_dict:
        occ_dict[item] = float(occ_dict[item])

    return(occ_dict)


def low_mem_setup(uvd_type, uvf_type, gpu_files, metafits_file, **kwargs):
    
    chan_list = [str(chan).zfill(2) for chan in range(1, 25)]

    init_str_list = [f"gpubox{chan}" for chan in chan_list[:3]]
    init_files = []
    for st in init_str_list:
        init_files += [path for path in gpu_files if st in path]

    print(f"init box files are {init_files}")
    uvd_obj = uvd_type()
    uvd_obj.read(init_files + metafits_file, **kwargs)
    uvf_obj = uvf_type(uvd_obj)

    for chan_group in range(1, 8):
        str_list = [f"gpubox{chan}" for chan in chan_list[3 * chan_group: 3 * (chan_group + 1)]]
        box_files = []
        for st in str_list:
            box_files += [path for path in gpu_files if st in path]

        print(f"box files for this iteration are {box_files}")
        uvd_obj = uvd_type()
        uvd_obj.read(box_files + metafits_file, **kwargs)
        uvf_obj.__add__(uvf_type(uvd_obj), axis="frequency", inplace=True)
        print(f"INS nfreqs is {uvf_obj.Nfreqs}")

    return(uvf_obj)


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
parser.add_argument('-s', '--start_flag', type=float, default=2.0,
                    help='The number of seconds to flag at the beginning of the obs.')
parser.add_argument('-e', '--end_flag', type=float, default=2.0,
                    help='The number of seconds to flag at the end of the obs.')
args = parser.parse_args()

gpu_files = [path for path in args.filelist if ".fits" in path]
mwaf_files = [path for path in args.filelist if ".mwaf" in path]
metafits_file = [path for path in args.filelist if ".metafits" in path]

ins = low_mem_setup(SS, INS, gpu_files, metafits_file, correct_cable_len=True,
                    phase_to_pointing_center=True, ant_str='cross', diff=True,
                    flag_choice='original', flag_init=True,
                    start_flag=args.start_flag, end_flag=args.end_flag)
prefix = f"{args.outdir}/{args.obsid}"
ins.write(prefix, clobber=True)

if args.rfi_flag:

    uvd = UVData()
    uvd.read(gpu_files + metafits_file, correct_cable_len=True,
             phase_to_pointing_center=True, ant_str='cross', read_data=False,
             flag_init=True)
    uvf = UVFlag(uvd, waterfall=True, mode='flag')

    num_init_flag = np.sum(ins.metric_array.mask)
    int_time = uvd.integration_time[0]
    print(f"Using int_time {int_time}")
    num_int_flag = (args.start_flag + args.end_flag) / int_time

    with open(f"{DATA_PATH}/MWA_EoR_Highband_shape_dict.yml", "r") as shape_file:
        shape_dict = yaml.safe_load(shape_file)
    sig_thresh = {shape: 5 for shape in shape_dict}
    sig_thresh["narrow"] = 5
    sig_thresh["streak"] = 10
    print(f"Flagging these shapes: {shape_dict}")

    mf = MF(ins.freq_array, sig_thresh, shape_dict=shape_dict, N_samp_thresh=20)
    mf.apply_match_test(ins, apply_samp_thresh=True)

    occ_dict = calc_occ(ins, num_init_flag, num_int_flag, shape_dict)
    with open(f"{prefix}_occ.yml", "w") as occ_file:
        yaml.safe_dump(occ_dict, occ_file)

    ins.write(prefix, output_type='mask', clobber=True)
    print(ins.Nfreqs)
    print(uvf.Nfreqs)
    ins.write(prefix, output_type='flags', uvf=uvf, clobber=True)
    ins.write(prefix, output_type='match_events', clobber=True)
    if args.write_mwaf:
        ins.write(prefix, output_type='mwaf', metafits_file=metafits_file,
                  mwaf_files=mwaf_files, Ncoarse=len(gpu_files))
