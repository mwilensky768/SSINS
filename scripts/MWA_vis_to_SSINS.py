from SSINS import INS, SS, MF
from SSINS.data import DATA_PATH
from pyuvdata import UVData, UVFlag
import yaml
import argparse
from astropy.io import fits
from astropy.time import Time
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
            occ_dict[event[2]] += ins.Npols / total_valid
        else:
            occ_dict[event[2]] += 1. / (ins.metric_array.shape[0] - num_int_flag)

    for item in occ_dict:
        occ_dict[item] = float(occ_dict[item])

    return(occ_dict)


def get_gpubox_map(gpu_files):
    chan_list = [str(chan).zfill(2) for chan in range(1, 25)]
    gpubox_map = {}
    for chan in chan_list:
        st = f"gpubox{chan}"
        gpubox_map[chan] = sorted([path for path in gpu_files if st in path])
    return gpubox_map


def find_gpubox_file_starts_ends(gpu_files):
    """MWA gpubox files don't necessarily start and end at the same time. pyuvdata
    does handle an observation correctly when *all* gpubox files are provided,
    but when only a few are provided (like the "low_mem_setup"), then unpacked
    data may not have the same dimensions. This function provides the start and
    end times for *all* gpuboxes, allowing the caller to remove data that
    shouldn't belong.

    """
    times = {}
    biggest_first = None
    smallest_last = None
    for chan, g in get_gpubox_map(gpu_files).items():
        headstart = fits.getheader(g[0], 1)
        headfin = fits.getheader(g[-1], -1)
        first_time = headstart["TIME"] + headstart["MILLITIM"] / 1000.0
        last_time = headfin["TIME"] + headfin["MILLITIM"] / 1000.0
        times[chan] = (first_time, last_time)

        if biggest_first is None:
            biggest_first = first_time
        elif biggest_first < first_time:
            biggest_first = first_time
        if smallest_last is None:
            smallest_last = last_time
        elif smallest_last > last_time:
            smallest_last = last_time
    times["first"] = biggest_first
    times["last"] = smallest_last
    return times


def low_mem_setup(uvd_type, uvf_type, gpu_files, metafits_file, **kwargs):
    times = find_gpubox_file_starts_ends(gpu_files)
    chan_list = [str(chan).zfill(2) for chan in range(1, 25)]

    init_files = []
    start_time = None
    end_time = None
    for chan in chan_list[:3]:
        init_files += [path for path in gpu_files if f"gpubox{chan}" in path]
        # Find the start time that pyuvdata will give to this subset of gpubox
        # files.
        t = times[chan]
        if start_time is None:
            start_time = t[0]
        elif start_time > t[0]:
            start_time = t[0]
        if end_time is None:
            end_time = t[1]
        elif end_time < t[1]:
            end_time = t[1]

    gpubox0_header = fits.getheader(gpu_files[0], 1)
    int_time = gpubox0_header["INTTIME"]

    time_array = np.arange(start_time + int_time / 2.0,
                           end_time + int_time / 2.0 + int_time,
                           int_time)
    jd_time_array = Time(time_array, format="unix", scale="utc").jd.astype(float)

    print(f"init box files are {init_files}")
    uvd_obj = uvd_type()
    uvd_obj.read(init_files + metafits_file, times=jd_time_array, **kwargs)
    uvf_obj = uvf_type(uvd_obj)

    for chan_group in range(1, 8):
        box_files = []
        for chan in chan_list[3 * chan_group: 3 * (chan_group + 1)]:
            box_files += [path for path in gpu_files if f"gpubox{chan}" in path]

        print(f"box files for this iteration are {box_files}")
        uvd_obj = uvd_type()
        uvd_obj.read(box_files + metafits_file, times=jd_time_array, **kwargs)
        uvf_obj = uvf_type(uvd_obj).__add__(uvf_obj, axis="frequency", inplace=False)
        assert np.all(uvf_obj.freq_array[1:] > uvf_obj.freq_array[:-1]), "Frequencies are out of order for uvf object."
        print(f"INS nfreqs is {uvf_obj.Nfreqs}")

    return(uvf_obj, jd_time_array)


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

ins, jd_times = low_mem_setup(SS, INS, gpu_files, metafits_file,
                              correct_cable_len=True, phase_to_pointing_center=True,
                              ant_str='cross', diff=True, flag_choice='original',
                              flag_init=True, )
prefix = f"{args.outdir}/{args.obsid}"
ins.write(prefix, clobber=True)

if args.rfi_flag:

    uvd = UVData()
    uvd.read(gpu_files + metafits_file, correct_cable_len=True,
             phase_to_pointing_center=True, ant_str='cross', read_data=False,
             flag_init=True, times=jd_times)
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
