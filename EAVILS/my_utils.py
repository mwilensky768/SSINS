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


def split_list_on_items(lst, items):
    result = []
    sublist = []
    for x in lst:
        if x in items:
            result.append(sublist)
            sublist = [x]
        else:
            sublist.append(x)
    result.append(sublist)
    return result


def unpack_indices(pos_list, shape, init_check=True):
    """
    A recursive function which takes a list of 1-d indices and an array shape and determines what
    multidimensional indices it would have if the array was in the given array shape. init_check
    is reserved for the recursive use of this function
    """
    pos_list = np.array(pos_list)
    if len(shape) == 1:
        indices = []
    else:
        pos_list, indices = unpack_indices(pos_list, shape[1:], init_check=False)

    current_index = pos_list % shape[0]
    pos_list = (pos_list - current_index) / shape[0]
    indices.append(current_index)

    if not init_check:
        return pos_list, indices
    return np.array(list(reversed(indices)), dtype=int)


def find_maxes(array, N, vals_only=False):
    """
    Identifies the maximum N values in a multidimensional array
    """
    save_array = copy.deepcopy(array)

    array = array.flatten()

    temp = np.argpartition(-array, N)

    result_args = temp[:N]

    arg_indices = unpack_indices(result_args, save_array.shape)

    max_vals = []
    for arg in result_args:
        max_vals.append(save_array.flatten()[arg])

    if vals_only:
        return max_vals
    return arg_indices, max_vals


def cust_cmap(name):
    if name == "suncycle":
        initial_map = mpl.cm.twilight(np.arange(256 * 2))
        cmap = np.array(
            [
                initial_map[:, 0] ** 0.8,
                initial_map[:, 1] ** 2,
                initial_map[:, 2] * 0.35,
                initial_map[:, 0],
            ]
        ).transpose()
        cmap = mpl.colors.ListedColormap(cmap, name=name, N=cmap.shape[0])

    if name == "miami":
        initial_map = mpl.cm.twilight(np.arange(256 * 2))
        cmap = np.array(
            [
                initial_map[:, 0] ** 0.5,
                initial_map[:, 1] ** 2,
                1 - initial_map[:, 2],
                initial_map[:, 0],
            ]
        ).transpose()
        cmap = mpl.colors.ListedColormap(cmap, name=name, N=cmap.shape[0])

    if name == "neonbow":
        rgb = color_maker(
            np.pi / 2, [0.75, -0.75, 0.25], [1, 1, 0.5], [1, 1, 1], [1, 1, 1], [1, 1, 2]
        ).rgb
        cmap = np.array(rgb).transpose()
        cmap = mpl.colors.ListedColormap(cmap, name=name, N=cmap.shape[0])

    if name == "smoothbow":
        rgb = color_maker(
            0.4, [0.65, 0, 0.35], f=[1, 1, 1], A=[1, 1, 0.8], B=[1, 1, 1], p=[1, 1, 0.5]
        ).rgb
        cmap = np.array(rgb).transpose()
        cmap = mpl.colors.ListedColormap(cmap, name=name, N=cmap.shape[0])

    return cmap


class color_maker:

    def __init__(
        self,
        global_phi=0,
        phi=[0, 0, 0],
        f=[1, 1, 1],
        A=[1, 1, 1],
        B=[1, 1, 1],
        p=[1, 1, 1],
        name="default",
    ):
        x = np.arange(0, 2 * np.pi, 2 * np.pi / 512)
        self.global_phi = global_phi * 2 * np.pi
        self.phi = phi
        self.phi = np.array(self.phi) * 2 * np.pi + self.global_phi
        self.f = f
        self.A = A
        self.B = B
        self.p = p
        self.rgb = [
            self.A[0]
            * (1 / 2 - self.B[0] * np.sin(self.f[0] * x - self.phi[0]) / 2)
            ** self.p[0],
            self.A[1]
            * (1 / 2 - self.B[1] * np.sin(self.f[1] * x - self.phi[1]) / 2)
            ** self.p[1],
            self.A[2]
            * (1 / 2 - self.B[2] * np.sin(self.f[2] * x - self.phi[2]) / 2)
            ** self.p[2],
            np.ones(512),
        ]
        self.name = name

    def rgb(self):
        return self.rgb


def data(uvd):
    return uvd.data_array.reshape(uvd.Ntimes, uvd.Nbls, uvd.Nfreqs, uvd.Npols)


def reader(
    obs_id,
    input_folder="",
    conjugate_baselines=False,
    split_autos=False,
    options={},
    use_ss_as_uvd=False,
    metafits_ant_check=True,
    antenna_position_flags_csv="",
    detect_time_cuts=True,extension='uvfits',input_cut_antennas=None
):

    time_cuts=None
    obs_id = str(obs_id)
    fits_file = obs_id + "."+extension
    read_file = os.path.join(input_folder, fits_file)
    print("reading ", read_file)
    if not use_ss_as_uvd:
        uvd = UVData.from_file(read_file, use_future_array_shapes=True)

    else:
        print("Reading in as a SSINS ss object")
        # Note: we are calling this object 'uvd' for code functionality, but it is in fact a ss object!
        uvd = SS()
        uvd.read(read_file, diff=False)
        
    if detect_time_cuts:
        flag_reshaped = uvd.flag_array.reshape((uvd.Ntimes,uvd.Nbls,uvd.Nfreqs,uvd.Npols))
        bad_time_flags = np.all(flag_reshaped,axis=(1,2,3))
        good_times = np.arange(len(bad_time_flags))[~bad_time_flags]
        time_cuts = (min(good_times),max(good_times+1))
        if not time_cuts[1]-time_cuts[0] ==len(good_times):
            raise Exception(f'Unreliable time flags found for {read_file}. Expecting only beginning and ending times to be flagged, not middle times. Total number of time indices: {uvd.Ntimes}. Flagged time indices: {np.arange(len(bad_time_flags))[bad_time_flags]}')
        else:
            print(f'Total number of time indices: {uvd.Ntimes}. Flagged time indices: {np.arange(len(bad_time_flags))[bad_time_flags]}. time_cuts set to {time_cuts}')
    if "time_cuts" in options.keys():
        time_cuts = options["time_cuts"]
        print(f'Options given. Overriding any existing time_cuts with {time_cuts}')
        
    if time_cuts is not None:
        print(f"trimming times to include indices between: {time_cuts}")
        # fmt: off
        uvd.select(times=np.unique(uvd.time_array)[time_cuts[0]:time_cuts[1]])

    if conjugate_baselines:
        print("conjugating bls")
        uvd.conjugate_bls(convention="v>0")
    if input_cut_antennas is None:
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
                ant_name.rstrip() for ant_name in uvd.antenna_names
            ]  # Gets rid of unnecessary whitespace
    
            ant_name_num_dict = {}
            for i, name in enumerate(antenna_names_fix):
                ant_num = uvd.antenna_numbers[i]
                ant_name_num_dict[name] = ant_num
    
            for tile in bad_tiles:
                cut_antennas.append(ant_name_num_dict[tile])
            print(f"Antennas found bad via {metafits_file_name}:", cut_antennas)

    
    if len(cut_antennas) > 0:

        keep_antennas = uvd.antenna_numbers
        keep_antennas = [ant for ant in keep_antennas if ant not in cut_antennas]
        uvd.select(antenna_nums=keep_antennas)

    if not split_autos:
        return uvd
    else:

        uvd_autos = uvd.copy()
        uvd_autos.select(ant_str="auto")
        uvd.select(ant_str="cross")

        

        return uvd, uvd_autos


def shape_dict(shape_name, add_subTV=False):
    if shape_name == "MWA":
        with open(
            f"{SSINS_data.DATA_PATH}/MWA_EoR_Highband_shape_dict.yml", "r"
        ) as shape_file:
            shape_dict = yaml.safe_load(shape_file)
            if add_subTV:
                shape_dict["subTV"] = [167075000.0, 174000000.0]
    return shape_dict


def tri_num(num, reverse=False):
    if not reverse:
        output = 0
        for i in range(num):
            output += i + 1
    else:
        output = 0
        while num > 0:
            output += 1
            num -= output
        if num < 0:
            print("Warning: not a true triangular number, returning nan")
            output = np.nan
    return output


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
        shape_dict = "MWA"
    if shape_dict == "MWA":
        with open(
            f"{SSINS_data.DATA_PATH}/MWA_EoR_Highband_shape_dict.yml", "r"
        ) as shape_file:
            shape_dict = yaml.safe_load(shape_file)
    # print(shape_dict)

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


def obs_id_convert(obs_id):
    output = astrotime.Time(obs_id, scale="utc", format="gps")

    output = (output.strftime("%m-%d-%Y"), output.strftime("%H:%M:%S"))

    return output


def recursive_search(folder, exclude_list=[]):
    files = {}
    local_files = []
    for item in os.listdir(folder):

        if len(item.split(".")) < 2 and item not in exclude_list:

            files[item] = recursive_search(os.path.join(folder, item))
        else:
            local_files.append(item)
    files[""] = local_files
    return files


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
    """im = ax.get_images()
    print(repr(im))
    print(type(im))
    print(len(im))"""
    """bbox = ax.get_window_extent()
    width, height = bbox.width, bbox.height"""

    init_aspect, disp_ratio = get_aspect(ax)

    # disp_ratio, data_ratio = get_aspect(ax)
    # disp_ratio, data_x, data_y = get_aspect(ax)

    # init_aspect = 1/data_ratio
    # init_aspect = 1/data_y
    # print(data_x,data_y, data_y/data_x)
    # extent =  im[0].get_extent()
    # ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)
    ax.set_aspect(aspect * init_aspect / disp_ratio)

    # ax.set_aspect(aspect)
