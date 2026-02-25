import numpy as np
import matplotlib as mpl
from astropy import units as u
import astropy.time as astrotime
import astropy.io.fits as fits
import os
import copy
import ast
from pyuvdata import UVData
from SSINS import SS
from SSINS import data as SSINS_data
import yaml
import csv


def mwa_pointings(az, alt, tolerance=0.01):
    # Order is Az, Alt
    pting_trans = {
        -3: (90, 69.1655),
        -2: (90, 76.2838),
        -1: (90, 83.1912),
        0: (None, 90),
        1: (270, 83.1912),
        2: (270, 76.2838),
        3: (270, 69.1655),
    }

    for pting in pting_trans.keys():
        test_az, test_alt = pting_trans[pting]
        if test_az is None:
            test_az = az
        if test_az - tolerance < az and az < test_az + tolerance:
            if test_alt - tolerance < alt and alt < test_alt + tolerance:
                return pting




def reader(
    obs_id,
    input_folder="",
    split_autos=False,
    metafits_ant_check=True,
    detect_time_cuts=True,
    extension='uvfits'
):

    time_cuts=None
    obs_id = str(obs_id)
    fits_file = obs_id + "."+extension
    read_file = os.path.join(input_folder, fits_file)

    
    print("Reading in ", read_file," as an undiffed SSINS ss object")
    ss = SS()
    ss.read(read_file, diff=False,flag_init=True)
        
    if detect_time_cuts:
        #Detects times flagged as bad. Will error if these are not the beginning and/or end times
        flag_reshaped = ss.flag_array.reshape((ss.Ntimes,ss.Nbls,ss.Nfreqs,ss.Npols))
        bad_time_flags = np.all(flag_reshaped,axis=(1,2,3))
        good_times = np.arange(len(bad_time_flags))[~bad_time_flags]
        time_cuts = (min(good_times),max(good_times+1))
        if not time_cuts[1]-time_cuts[0] ==len(good_times):
            raise Exception(f'Unreliable time flags found for {read_file}. Expecting only beginning and ending times to be flagged, not middle times. Total number of time indices: {ss.Ntimes}. Flagged time indices: {np.arange(len(bad_time_flags))[bad_time_flags]}')
        else:
            print(f'Total number of time indices: {ss.Ntimes}. Flagged time indices: {np.arange(len(bad_time_flags))[bad_time_flags]}. time_cuts set to {time_cuts}')

        
    if time_cuts is not None:
        print(f"trimming times to include indices between: {time_cuts}")
        # fmt: off
        ss.select(times=np.unique(ss.time_array)[time_cuts[0]:time_cuts[1]])

    

    cut_antennas = []
    if metafits_ant_check:
        metafits_file_name = os.path.join(input_folder, f"{obs_id}.metafits")
        metafits = fits.open(metafits_file_name)

        bad_tiles = []
        # Metafits files save flags, TileNames etc in pairs of polarizations, so we index across pairs here
        for ind in range(len(metafits["TILEDATA"].data.field("flag")) // 2):
            
            if sum(metafits["TILEDATA"].data.field("flag")[(ind * 2):(ind * 2 + 2)]) > 0:
                bad_tiles.append(metafits["TILEDATA"].data.field("TileName")[ind * 2])

        antenna_names_fix = [
            ant_name.rstrip() for ant_name in ss.telescope.antenna_names
        ]  # Gets rid of unnecessary whitespace

        ant_name_num_dict = {}
        for i, name in enumerate(antenna_names_fix):
            ant_num = ss.telescope.antenna_numbers[i]
            ant_name_num_dict[name] = ant_num

        for tile in bad_tiles:
            cut_antennas.append(ant_name_num_dict[tile])
        print(f"Antennas found bad via {metafits_file_name}:", cut_antennas)

    
    if len(cut_antennas) > 0:

        keep_antennas = ss.antenna_numbers
        keep_antennas = [ant for ant in keep_antennas if ant not in cut_antennas]
        ss.select(antenna_nums=keep_antennas)

    if not split_autos:
        return ss
    else:

        ss_autos = ss.copy()
        ss_autos.select(ant_str="auto")
        ss.select(ant_str="cross")

        

        return ss, ss_autos


def get_shape_dict(shape_name, add_subTV=False):
    if shape_name == "MWA_high":
        with open(
            f"{SSINS_data.DATA_PATH}/MWA_EoR_Highband_shape_dict.yml", "r"
        ) as shape_file:
            shape_dict = yaml.safe_load(shape_file)
            if add_subTV:
                shape_dict["subTV"] = [167075000.0, 174000000.0]
    return shape_dict




def closest(lst, K, ineq=None):
    """
    Takes an argument of a list and a value, K, and returns the list's index of the item closest to the value.
    ineq can be 'g' for greater than or equal to, 'l' for less than or equal to,
    or None to just go to the closest value
    """

    lst = np.asarray(lst)
    if isinstance(K, type(1 * u.Hz)):
        K = K.value

    check_list = lst - K
    idx = np.abs(check_list).argmin()

    if lst[idx] < K and ineq == "g":
        if not (sorted(lst) == lst).all():
            raise ValueError(
                "To use a ineq argument other than None the input array must be sorted smallest to largest"
            )
        idx += 1
        if idx > len(lst) - 1:
            raise ValueError(
                "Largest value of array = "
                + str(lst[-1])
                + ", which is smaller than specified search value. Remove ineq='g' or choose smaller value"
            )

    elif lst[idx] > K and ineq == "l":
        if not (sorted(lst) == lst).all():
            raise ValueError(
                "To use a ineq argument other than None the input array must be sorted smallest to largest"
            )
        idx -= 1
        if idx < 0:
            raise ValueError(
                "Smallest value of array = "
                + str(lst[0])
                + ", which is larger than specified search value. Remove ineq='l' or choose larger value"
            )

    return idx


def freq_ind_finder(freqs, ranges, strict=False):
    # ranges should have format [freq1,[freq_start1,freq_end2],freq2,freq3,etc]
    indices = []

    warning_count_strict = 0

    if strict:
        strict_lower = "g"
        strict_upper = "l"
    else:
        strict_lower = None
        strict_upper = None

    for item in ranges:

        if isinstance(item, (list, tuple)):
            if len(item) != 2:
                raise IndexError("Freq ranges must contain exactly 2 numbers")
            elif item[0] >= item[1]:
                raise ValueError(
                    "Freq ranges must be ordered from smallest to largest, "
                    + str(item[0])
                    + " >= "
                    + str(item[1])
                )

            lower = closest(freqs, item[0], ineq=strict_lower)
            upper = closest(freqs, item[1], ineq=strict_upper)
            indices = indices + list(np.arange(lower, upper + 1, 1))

        else:
            if strict and warning_count_strict == 0:
                print(
                    "Warning: 'strict' keyword only affects frequency ranges, not individual frequencies"
                )
                warning_count_strict += 1
            indices.append(closest(freqs, item))

    return indices


def make_freq_mask(freqs, ranges, strict=True, return_list=False, shape_dict=None):
    if shape_dict is None:
        print("No shape_dict given, defaulting to the MWA shape_dict")
        shape_dict = "MWA_high"
    shape_dict = get_shape_dict(shape_dict)


    for item in ranges:
        if isinstance(item, list):
            for subitem in item:
                if not isinstance(subitem, type(1 * u.Hz)):
                    print(
                        "Warning: please ensure the ranges arguments of make_freq_mask contains "
                        "frequencies or pairs of frequencies and not indices"
                    )
                    break
        elif not isinstance(item, type(1 * u.Hz)):
            print(
                "Warning: please ensure the ranges arguments of make_freq_mask contains frequencies or pairs of \
frequencies and not indices"
            )
            break

    if isinstance(ranges, str):
        if ranges == "all":
            ranges = [[np.min(freqs), np.max(freqs)]]
        elif ranges in shape_dict.keys():
            ranges = [shape_dict[ranges]]
        else:
            print("There is no key in the shapes dict matching the input:", ranges)
            raise ValueError()

    freq_mask = np.zeros(len(freqs), dtype=bool)
    indices = freq_ind_finder(freqs, ranges, strict)

    for index in indices:
        freq_mask[index] = 1

    if return_list:
        freq_list = freqs[freq_mask]
        return freq_mask, freq_list
    else:
        return freq_mask


def coarse_band_flagging(
    Nfreqs=384,
    coarse_band_count=24,
    flag_centers=True,
    flag_edges=True,
    Ntimes=None,
    Npols=None,
    bad_edge_channel_fraction=1 / 8,
):
    # bad_edge_channel_fraction describes what fraction of the coarse band edge is considered bad.
    coarse_band_channel_size = int(Nfreqs // coarse_band_count)

    # Edge size is the number of channels above and below each coarse band threshold that should be cut.
    # Note it always will cut at least one
    edge_size = int(coarse_band_channel_size * bad_edge_channel_fraction // 2)
    if edge_size < 1:
        edge_size = 1

    edge_channel_distances = np.arange(0, edge_size)

    coarse_band_flags_ind = []
    for channel in range(Nfreqs):
        # Flags edges of coarse bands
        if flag_edges:

            if (
                0 in (channel + 1 + edge_channel_distances) % coarse_band_channel_size
                or 0 in (channel - edge_channel_distances) % coarse_band_channel_size
            ):
                coarse_band_flags_ind.append(channel)

        # Flags centers of coarse bands
        if flag_centers:
            if (channel + coarse_band_channel_size / 2) % coarse_band_channel_size == 0:
                coarse_band_flags_ind.append(channel)

    coarse_band_flags = np.zeros(Nfreqs, dtype=bool)
    for ind in coarse_band_flags_ind:
        coarse_band_flags[ind] = True
    if isinstance(Ntimes, type(None)) and isinstance(Npols, type(None)):
        return coarse_band_flags
    elif not isinstance(Ntimes, type(Npols)):
        print("error: if supplying Ntimes, must also supply Npols")
        raise TypeError()
    else:
        coarse_band_flags_array = np.zeros((Ntimes, Nfreqs, Npols), dtype=bool)
        for ti in range(Ntimes):
            for fi in range(Nfreqs):
                for pi in range(Npols):
                    coarse_band_flags_array[ti, fi, pi] = coarse_band_flags[fi]

        return coarse_band_flags_array


def get_aspect(ax):
    from operator import sub

    # Total figure size
    figW, figH = ax.get_figure().get_size_inches()
    # Axis size on figure
    _, _, w, h = ax.get_position().bounds
    # Ratio of display units
    disp_ratio = (figH * h) / (figW * w)
    # print('disp',disp_ratio)

    # Ratio of data units
    # Negative over negative because of the order of subtraction

    data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())
    # print('data',data_ratio)
    return disp_ratio / data_ratio, disp_ratio
    # return disp_ratio, np.abs(sub(*ax.get_xlim())), np.abs(sub(*ax.get_ylim()))




def forceAspect(ax, aspect=1):
    init_aspect, disp_ratio = get_aspect(ax)
    ax.set_aspect(aspect * init_aspect / disp_ratio)



def pad_by(bool_array,pad_by_count,axis=0,pad_beginning_bool=True,spread_flags_count=1):

    empty_pad = np.array([[0,0] for dim in range(len(bool_array.shape))])
    end_pad = deepcopy(empty_pad)
    begin_pad = deepcopy(empty_pad)
    
    end_pad[axis,1] = 1
    begin_pad[axis,0] = 1

    for iterations in range(pad_by_count):
        if iterations<spread_flags_count:
            bool_array = np.logical_or(
                        np.pad(bool_array, pad_width=begin_pad, 
                            mode='constant', constant_values=False),
                        np.pad(bool_array, pad_width=end_pad, 
                            mode='constant', constant_values=False)
                        )
        else:
            if pad_beginning_bool:
                bool_array = np.pad(bool_array, pad_width=begin_pad, 
                            mode='edge')
            else:
                bool_array = np.pad(bool_array, pad_width=end_pad, 
                            mode='edge')
    return bool_array