from SSINS import INS, SS, MF, util, Catalog_Plot as cp
from SSINS.data import DATA_PATH
from pyuvdata import UVData, UVFlag
import yaml
import argparse
from astropy.io import fits
from astropy.time import Time
import numpy as np


def get_agreeable_times(boxfiles, metafits_file):
    uvd = UVData()

    time_set = set()
    # Only grab times that everyone agrees exist
    for boxfile in boxfiles:
        uvd.read([boxfile, metafits_file], read_data=False)
        # initialize if it is the first iteration
        if len(time_set) == 0:
            time_set = set(np.unique(uvd.time_array))
        else:
            time_set.intersection_update(np.unique(uvd.time_array))

    return(time_set)


def low_mem_setup(uvd_type, uvf_type, gpu_files, metafits_file, **kwargs):

    chan_list = [str(chan).zfill(2) for chan in range(1, 25)]
    init_files = []
    for chan in chan_list[:3]:
        init_files += [path for path in gpu_files if f"gpubox{chan}" in path]

    times = get_agreeable_times(gpu_files, metafits_file[0])

    print(f"init box files are {init_files}")
    uvd_obj = uvd_type()
    uvd_obj.read(init_files + metafits_file, times=times, **kwargs)
    uvf_obj = uvf_type(uvd_obj)

    for chan_group in range(1, 8):
        box_files = []
        for chan in chan_list[3 * chan_group: 3 * (chan_group + 1)]:
            box_files += [path for path in gpu_files if f"gpubox{chan}" in path]

        print(f"box files for this iteration are {box_files}")
        uvd_obj = uvd_type()
        uvd_obj.read(box_files + metafits_file, times=times, **kwargs)
        uvf_obj = uvf_type(uvd_obj).__add__(uvf_obj, axis="frequency", inplace=False)
        assert np.all(uvf_obj.freq_array[1:] > uvf_obj.freq_array[:-1]), "Frequencies are out of order for uvf object."
        print(f"INS nfreqs is {uvf_obj.Nfreqs}")

    return(uvf_obj, times)


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
parser.add_argument('-p', '--plot', action='store_true',
                    help='Plot the INS object')
args = parser.parse_args()

gpu_files = [path for path in args.filelist if ".fits" in path]
mwaf_files = [path for path in args.filelist if ".mwaf" in path]
metafits_file = [path for path in args.filelist if ".metafits" in path]

ins, times = low_mem_setup(SS, INS, gpu_files, metafits_file,
                           correct_cable_len=True, phase_to_pointing_center=True,
                           ant_str='cross', diff=True, flag_choice='original',
                           flag_init=True)
prefix = f"{args.outdir}/{args.obsid}"
ins.write(prefix, clobber=True)

if args.plot:
    cp.INS_plot(ins, prefix, file_ext='pdf')

if args.rfi_flag:

    uvd = UVData()
    uvd.read(gpu_files + metafits_file, correct_cable_len=True,
             phase_to_pointing_center=True, ant_str='cross', read_data=False,
             flag_init=True, times=times)
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

    mf = MF(ins.freq_array, sig_thresh, shape_dict=shape_dict, tb_aggro=0.4)
    mf.apply_match_test(ins, time_broadcast=True)

    occ_dict = util.calc_occ(ins, mf, num_init_flag, num_int_flag=num_int_flag,
                             lump_narrowband=True)
    with open(f"{prefix}_occ.yml", "w") as occ_file:
        yaml.safe_dump(occ_dict, occ_file)

    ins.write(prefix, output_type='mask', clobber=True)
    ins.write(prefix, output_type='flags', uvf=uvf, clobber=True)
    ins.write(prefix, output_type='match_events', clobber=True)
    if args.write_mwaf:
        ins.write(prefix, output_type='mwaf', metafits_file=metafits_file,
                  mwaf_files=mwaf_files, Ncoarse=len(gpu_files))
