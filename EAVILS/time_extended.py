#!/Users/elillesk/miniconda3/envs/my_env/bin/python

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from matplotlib import colors
import argparse
import os
import copy
import astropy.time as time
import astropy.io.fits as fits
from astropy.coordinates import Angle
from astropy import units as u
import yaml

import my_utils
import vis_plotting as vis_plt


from SSINS import INS

from datetime import date

today = date.today()
formatted_date = today.strftime("%d-%m-%Y")

yaml_file_name = ""
info_file_dict = {}

"""
- data_folders should be given a list of the folders with h5 files needed to actually make the plots.
- metafits_folders takes a list of folders containing metafits files; needed to get pointing information.
- obs_id_list_file takes a text list of obs_ids :).
- split_pointings splits plots into roughly printable shapes, with pointings generally split into two plots.
  If this is left as fault, code will default to making one big plot for everything.
- max_obs_page just sets the maximum observations allowed per page when split_pointings is used, otherwise does nothing
- plot_pols_together simply designates whether pols should be grouped or metrics should be grouped in plot columns
- out_folder by default just automatically generates an output folder with the date,
  but can be given a different argument if desired
- skip_autos just skips making extended plots for the auto spectrum, only doing the cross spectrum
- yaml_file_name having an argument other than '' activates the info panel to be added to the plot.
  For now, this is not a flexible feature and should only be used if you're willing to get into the
  weeds and adapt the code. It was designed for a specific use case, with a specific handful of yaml
  files in mind, and will not work unless those are used
- info_file_dict has a similar function to yaml_file_name, with all the same caveats
- add_line_plots adds averages of EAVILS values across time for both pols as well as diffs of that.
  They aren't the most helpful but can give some sense of scale for how large peaks are when they
  blow out the color scale. Forced to False when split_pointings is set to True

Reciever data specific options:
- receiver_data_usage should generally be left false unless trying to look at receiver-specific data
- bypass_full_array should only be set to true if you only want to look at receiver-specific data
- receiver_choice is only useful when

"""


def main(
    data_folders,
    metafits_folders,
    obs_id_list_file,
    split_pointings=False,
    max_obs_page=8,
    plot_pols_together=True,
    out_folder=f"time_ext_out_{formatted_date}",
    skip_autos=True,
    yaml_file_name=yaml_file_name,
    add_line_plots=False,
    info_file_dict=info_file_dict,
    receiver_data_usage=False,
    bypass_full_array=False,
    receiver_choice="reciever12",
):

    if split_pointings:
        add_line_plots = False
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    print(f"Data folder: {data_folders}")
    print(f"Metafits folder: {metafits_folders}")
    print(f"Receiver data usage: {receiver_data_usage}")
    print(f"Bypass processing full array data: {bypass_full_array}")
    print(f"Split plots: {split_pointings}")

    # Sets up initial parameters and variables
    if skip_autos:
        auto_bool_list = [False]
    else:
        auto_bool_list = [False, True]

    with open(obs_id_list_file, "r") as file:
        print("reading", obs_id_list_file)
        obs_id_list = file.read().split()

    # If split_pointings is true, this chunk will generate a set of sub_lists based on pointing and max_obs_page.
    # Otherwise will group all obs_ids into one list
    # Obs_ids must be both in the input list file and have associated metafits to be included in the output lists
    if split_pointings:
        pointing_change_obs_ids = raw_pointing_list(metafits_folders, obs_id_list)
        sub_lists = []
        for ls in my_utils.split_list_on_items(obs_id_list, pointing_change_obs_ids):

            if len(ls) > max_obs_page:
                sub_lists.append(ls[:max_obs_page])
                sub_lists.append(ls[max_obs_page:])
            else:
                sub_lists.append(ls)
    else:
        full_obs_id_list = []
        for metafits_folder in metafits_folders:
            for file in os.listdir(metafits_folder):
                if ".metafits" in file:
                    full_obs_id_list.append(file.split(".")[0])
        obs_id_list = [obs_id for obs_id in full_obs_id_list if obs_id in obs_id_list]
        sub_lists = [obs_id_list]

    for obs_list_ind, obs_id_sub_list in enumerate(sub_lists):

        #############################################################################
        # Receiver plotting starts here
        #############################################################################
        rec_files = []
        for data_folder in data_folders:

            rec_files += [
                os.path.join(data_folder, f)
                for f in os.listdir(data_folder)
                if (
                    (f.split(".")[1] == "h5" and ("receiver" in f or "reciever" in f))
                    and (f.split("_")[0] in obs_id_sub_list)
                )
            ]

        #######################################################################################################
        # This section deals with making plots to look at individual receivers and antennas.
        # It's not useful for other things so can be ignored. Not actively maintained so no guarantee of usability.
        if receiver_data_usage and len(rec_files) > 0:

            rec_auto_files = []
            rec_cross_files = []
            for file in rec_files:

                if "auto" in file:
                    rec_auto_files.append(file)
                elif "cross" in file:
                    rec_cross_files.append(file)

            for auto_bool in auto_bool_list:
                if not auto_bool:
                    rec_file_list = rec_cross_files
                    auto_tag = "Cross"
                else:
                    rec_file_list = rec_auto_files
                    auto_tag = "Autos"
                print(auto_tag)
                if len(rec_file_list) == 0:
                    break
                (
                    rec_data_dict,
                    rec_filled_time_array,
                    rec_filled_obs_id_array,
                    rec_filled_lst_array,
                    rec_attr_dict,
                ) = h5_file_collector(rec_file_list, average_across=[])

                (
                    ra_list,
                    dec_list,
                    alt_list,
                    az_list,
                    pointing_change_results,
                    p_c_time_indices,
                ) = pointing_change_identifier(
                    metafits_folders, rec_filled_obs_id_array
                )

                yticks = obs_id_change_identifier(rec_filled_obs_id_array)

                time_names = rec_filled_obs_id_array
                unique_obs_ids = ["%i" % (int(time_names[tick])) for tick in yticks]
                right_yticklabels = [
                    (
                        f'{"{:02d}".format(int(Angle(rec_filled_lst_array[tick], u.radian).hms.h))}:'
                        f'{"{:02d}".format(int(Angle(rec_filled_lst_array[tick], u.radian).hms.m))}'
                    )
                    for tick in yticks
                ]

                stitched_data = None

                filled_data = rec_data_dict[receiver_choice]
                for i in range(len(p_c_time_indices) - 1):
                    begin_ind = p_c_time_indices[i]
                    end_ind = p_c_time_indices[i + 1]

                    mean_sub = np.abs(
                        filled_data[begin_ind:end_ind, :, :, :]
                    ) - np.nanmean(
                        np.abs(filled_data[begin_ind:end_ind, :, :, :]), axis=0
                    )
                    if stitched_data is None:
                        stitched_data = mean_sub
                    else:
                        stitched_data = np.append(stitched_data, mean_sub, axis=0)

                pols = ["XX", "YY", "XY", "YX"]

                freq_array = rec_attr_dict["freq_array"]

                row_count = rec_attr_dict["Nbls"]
                col_count = rec_attr_dict["Npols"]

                ant_labels = rec_attr_dict["ant_labels"]
                height_factor = len(stitched_data) / 1000
                fig, axs = plt.subplots(
                    nrows=row_count,
                    ncols=col_count,
                    figsize=(col_count * 4, row_count * 8),
                    dpi=300,
                )

                xticks = np.arange(12, len(freq_array), 50)
                xticklabels = [
                    "%.0f" % (freq_array[tick] * 10 ** (-6)) for tick in xticks
                ]

                date_string = time.Time(
                    rec_filled_time_array[0], scale="utc", format="jd"
                ).strftime("%m-%d-%Y")

                for pol_ind in range(len(pols)):
                    for ind in range(row_count):

                        ax = axs[ind, pol_ind]
                        ant_ind = ind

                        active_plot = ax.imshow(
                            stitched_data[:, ant_ind, :, pol_ind],
                            aspect=1,
                            cmap="coolwarm",
                        )  # ,vmin=-500,vmax=500)

                        fig.colorbar(active_plot, orientation="vertical")
                        ax.set_xticks(xticks)
                        ax.set_xticklabels(xticklabels, fontsize=6)
                        ax.set_yticks(yticks)
                        ax.set_yticklabels(unique_obs_ids, fontsize=6)
                        ax.set_title(
                            f"{pols[pol_ind]}, ant index: {str(ant_labels[ant_ind])}",
                            fontsize=8,
                        )
                if split_pointings:
                    label = obs_list_ind
                else:
                    label = "all"
                save_name = f"{date_string}_{receiver_choice}_plot{label}.pdf"
                save_name = os.path.join(out_folder, save_name)
                fig.savefig(save_name, bbox_inches="tight")
                plt.close(fig)
        ##########################################################################
        # End of receiver section

        ###########################################################################################
        # Full array plotting starts here!
        ##########################################################################################
        full_array_files = []
        for data_folder in data_folders:
            full_array_files += [
                os.path.join(data_folder, f)
                for f in os.listdir(data_folder)
                if (
                    (
                        f.split(".")[1] == "h5"
                        and not ("receiver" in f or "reciever" in f)
                    )
                    and (f.split("_")[0] in obs_id_sub_list)
                )
            ]
        if not bypass_full_array and len(full_array_files) > 0:

            auto_files = []
            cross_files = []
            for file in full_array_files:
                if "auto" in file:
                    auto_files.append(file)
                elif "cross" in file:
                    cross_files.append(file)

            if yaml_file_name != "":
                add_info_column = True
            elif info_file_dict != {}:
                add_info_column = True
            else:
                add_info_column = False

            for auto_bool in auto_bool_list:
                pols = ["XX", "YY", "XY", "YX", ""]

                if not auto_bool:

                    active_file_list = cross_files
                    auto_tag = "Cross"
                else:

                    active_file_list = auto_files
                    auto_tag = "Autos"
                print(auto_tag)
                if len(active_file_list) == 0:
                    continue

                #####################################
                # Sets up figure data              #
                ###################################
                (
                    active_data_dict,
                    active_filled_time_array,
                    active_filled_obs_id_array,
                    active_filled_lst_array,
                    active_attr_dict,
                ) = h5_file_collector(
                    active_file_list,
                    average_across=["blmean"],
                    metafits_folders=metafits_folders,
                    use_cross_array=False,
                    add_ssins_mask=not auto_bool,
                )

                #####################################
                # Identifies pointing changes      #
                ###################################
                (
                    ra_list,
                    dec_list,
                    alt_list,
                    az_list,
                    pointing_change_results,
                    p_c_time_indices,
                ) = pointing_change_identifier(
                    metafits_folders, active_filled_obs_id_array
                )

                # Sets up figure data and settings
                EAVILS = (
                    np.abs(active_data_dict["blmean"]) - active_data_dict["time_mean"]
                ) / (active_data_dict["stdv_array"] / np.sqrt(active_attr_dict["Nbls"]))
                if auto_bool:
                    SSINS = active_data_dict["metric_ms"]
                else:
                    SSINS = active_data_dict["metric_ms_best_avail"]
                if not auto_bool:
                    raster_data_list = [EAVILS, SSINS, active_data_dict["SSINS_mask"]]
                    raster_data_titles = ["EAVILS", "SSINS", "SSINS mask"]
                if auto_bool:
                    raster_data_list = [EAVILS, SSINS]
                    raster_data_titles = ["EAVILS", "SSINS"]
                raster_vlims = [(-5, 5), (-5, 5), (0, 1)]

                raster_cmaps = ["coolwarm", "coolwarm", "Greens"]

                if plot_pols_together:
                    # Combinations of data index and pol index
                    # (e.g. [(data_ind_A,pol_ind_A),(data_ind_B,pol_ind_B),...])
                    raster_data_pol_inds = [(0, 0), (1, 0), (0, 1), (1, 1)]
                else:
                    raster_data_pol_inds = [(0, 0), (0, 1), (1, 0), (1, 1)]

                # adds data and pol index for the SSINS mask if applicable
                if not auto_bool:
                    raster_data_pol_inds.append((2, None))

                info_titles = ["Info"]

                if add_line_plots:
                    lineplot_data_list = [
                        active_data_dict["time_mean"],
                        active_data_dict["time_mean"],
                    ]
                    lineplot_data_titles = [
                        "Mean(abs(V))\nover t & bl",
                        "Mean(abs(V))\nover t & bl\n obsid diff",
                    ]
                    lineplot_data_diff_bool = [False, True]

                else:
                    lineplot_data_list = []
                    lineplot_data_titles = []
                    lineplot_data_diff_bool = []

                raster_col_count = len(raster_data_pol_inds)
                lineplot_col_count = len(lineplot_data_list)
                if add_info_column:
                    info_col_count = 1
                else:
                    info_col_count = 0
                col_count = raster_col_count + info_col_count + lineplot_col_count

                row_count = 1

                left_buffer = 0.2
                right_buffer = 0.2
                col_width = (1 - left_buffer - right_buffer) / col_count

                height_factor = len(active_filled_time_array) / 100
                size_factor = 2

                fig, axs = plt.subplots(
                    nrows=row_count,
                    ncols=col_count,
                    figsize=(
                        col_count * 1.2 * size_factor,
                        row_count * height_factor * size_factor,
                    ),
                    dpi=300,
                )

                freq_array = active_attr_dict["freq_array"]
                xticks = np.arange(12, len(freq_array), 100)
                xticklabels = [
                    "%.0f" % (freq_array[tick] * 10 ** (-6)) for tick in xticks
                ]

                yticks = obs_id_change_identifier(active_filled_obs_id_array)

                time_names = active_filled_obs_id_array
                unique_obs_ids = ["%i" % (int(time_names[tick])) for tick in yticks]
                # fmt: off
                right_yticklabels = [
                    (f"{int(Angle(active_filled_lst_array[tick],u.radian).hms.h) :02d}:"
                     f"{int(Angle(active_filled_lst_array[tick],u.radian).hms.m):02d}")
                    for tick in yticks
                ]

                date_string = time.Time(
                    active_filled_time_array[0], scale="utc", format="jd"
                ).strftime("%m-%d-%Y")

                plt.subplots_adjust(wspace=0, hspace=0)

                #####################################
                # Creates raster plots             #
                ###################################

                ax_index = 0
                first_column_bool = True
                for data_ind, pol_ind in raster_data_pol_inds:

                    data = raster_data_list[data_ind]

                    data = np.ma.array(
                        data,
                        mask=my_utils.coarse_band_flagging(
                            Nfreqs=data.shape[1],
                            coarse_band_count=24,
                            flag_edges=True,
                            Ntimes=data.shape[0],
                            Npols=data.shape[2],
                        ),
                    )

                    ax = axs[ax_index]
                    aspect_adjust = data.shape[1] / active_attr_dict["Nfreqs"]

                    vmin, vmax = raster_vlims[data_ind]
                    if pol_ind is None:
                        temp_pol_ind = -1
                        # In the case where pol_ind is not directly applicable uses -1 as pol ind.
                        # Possible could have unintended effects if things change but should work
                        # well for current design (hacky)
                    else:
                        temp_pol_ind = pol_ind
                    active_plot = ax.imshow(
                        data[:, :, temp_pol_ind],
                        cmap=raster_cmaps[data_ind],
                        vmin=vmin,
                        vmax=vmax,
                        aspect=3.55 * aspect_adjust,
                        interpolation=None,
                    )
                    xticks_mod = data.shape[1] / len(freq_array)
                    ax.set_xticks(xticks * xticks_mod)

                    ax.set_xlabel("freq (MHz)", fontsize=1 * size_factor)
                    ax.set_xticklabels(xticklabels, fontsize=2 * size_factor)

                    if first_column_bool:
                        ax.set_yticks(yticks)
                        ax.set_yticklabels(unique_obs_ids, fontsize=3 * size_factor)
                        ax.tick_params(axis="y", labelrotation=90)
                        ax.set_ylabel("OBS ID")
                        first_column_ax = ax

                    else:
                        ax.set_yticks([])
                    ax.set_position(
                        [col_width * ax_index + left_buffer, 0, col_width, 0.9]
                    )

                    prev_ax_pos = ax.get_position()
                    if first_column_bool:
                        left_col_pos = prev_ax_pos
                    ax.set_title(
                        f"{pols[temp_pol_ind]}, {raster_data_titles[data_ind]}",
                        fontsize=6 * size_factor,
                    )

                    # Adds printed max to plots
                    if raster_data_titles[data_ind] == "EAVILS":
                        for loc in yticks:

                            nanrow_bool = True
                            min_t_ind = loc
                            search_t_ind = loc
                            while nanrow_bool:
                                nanrow_bool = np.any(
                                    ~np.isnan(data[search_t_ind, :, pol_ind])
                                )
                                if search_t_ind < len(data) - 1:
                                    search_t_ind += 1
                                else:
                                    nanrow_bool = False
                            max_t_ind = search_t_ind

                            max_z_score = np.nanmax(
                                data[min_t_ind:max_t_ind, :, pol_ind]
                            )

                            max_z_score = np.round(max_z_score, 1)
                            ax.text(
                                20,
                                loc - 2.5,
                                f"{max_z_score}",
                                fontsize=4.6 * size_factor,
                                verticalalignment="bottom",
                                bbox=dict(
                                    alpha=0.3, color="green", mutation_aspect=0.5
                                ),
                            )

                    first_column_bool = False
                    ax_index += 1

                #####################################
                # Adds info column                 #
                ###################################

                if add_info_column:
                    if yaml_file_name != "":
                        with open(yaml_file_name, "r") as yaml_file:
                            meta_dict = yaml.safe_load(yaml_file)
                    if info_file_dict != {}:
                        label_dict = {}
                        for label in info_file_dict.keys():
                            file_name = info_file_dict[label]["file"]
                            label_color = info_file_dict[label]["color"]
                            with open(file_name, "r") as file:
                                obslist = file.read().split("\n")
                                for obsid in obslist:
                                    label_dict[obsid] = label

                    ax = axs[ax_index]
                    info_array = np.full(
                        (len(active_filled_obs_id_array), active_attr_dict["Nfreqs"]),
                        np.nan,
                    )

                    ax.imshow(info_array, aspect=3.55)
                    ax.set_position(
                        [col_width * ax_index + left_buffer, 0, col_width, 0.9]
                    )

                    for obs_id, loc in zip(unique_obs_ids, yticks):
                        text_info = ""
                        edgecolor = "white"
                        if yaml_file_name != "":
                            this_dict = copy.deepcopy(meta_dict[obs_id])

                            present_rfi = []

                            rfi_bool = False
                            for key in this_dict[yaml_type].keys():

                                if key == "total":
                                    pass
                                elif this_dict[yaml_type][key] == 0:
                                    # present_rfi.append('___')
                                    present_rfi.append("   ")
                                else:
                                    present_rfi.append(key[:3])
                                    rfi_bool = True

                            if yaml_type == "SSINS":
                                limit_tag = ""
                                limit_bool = True
                                if not this_dict["limit_incl"]:
                                    limit_tag = "limit EXCLUDED"
                                    limit_bool = False

                                wall_of_shame_tag = ""
                                wall_of_shame_bool = False
                                if this_dict["wall_of_shame"]:
                                    wall_of_shame_tag = "WALL OF SHAME"
                                    wall_of_shame_bool = True

                                iono_cut_tag = ""
                                iono_cut_bool = not this_dict["aws"]
                                if iono_cut_bool:
                                    iono_cut_tag = "iono_cut"

                            text_info += str(present_rfi)
                            text_info = text_info.replace(",", " ")
                            text_info = text_info.replace("'", "")
                            text_info = text_info.replace("[", "")
                            text_info = text_info.replace("]", "")
                            text_info += "\n\n"
                            if yaml_type == "SSINS":
                                text_info += limit_tag
                                text_info += "\n\n"
                                text_info += wall_of_shame_tag
                                text_info += "\n\n"
                                text_info += iono_cut_tag

                                color_choice = None
                                if limit_bool:
                                    color_choice = "lightgray"

                                if wall_of_shame_bool:
                                    color_choice = "yellowgreen"

                                if not iono_cut_bool and rfi_bool:
                                    color_choice = "red"

                                if iono_cut_bool and rfi_bool:
                                    color_choice = "purple"

                                if iono_cut_bool and not rfi_bool:
                                    color_choice = "cornflowerblue"

                                if color_choice is None:
                                    print(
                                        f"no matching color condition found for obs_id {obs_id}"
                                    )
                                    raise Exception

                                if iono_cut_bool:
                                    edgecolor = "red"
                                else:
                                    edgecolor = "white"
                            if yaml_type == "EAVILS_flag_frac":
                                if rfi_bool:
                                    edgecolor = "black"

                            extra_size = 1

                        if info_file_dict != {}:
                            try:
                                label = label_dict[obs_id]
                                text_info += label
                                color_choice = info_file_dict[label]["color"]
                            except KeyError:
                                text_info += ""
                                color_choice = "cornflowerblue"

                            extra_size = 1

                        fc = colors.to_rgba(color_choice)
                        fc = fc[:-1] + (0.3,)

                        ec = colors.to_rgba(edgecolor)

                        ax.text(
                            40,
                            loc + 5,
                            text_info,
                            fontsize=3 * size_factor * extra_size,
                            verticalalignment="top",
                            bbox=dict(facecolor=fc, edgecolor=ec),
                        )

                        ax.set_xticks([])
                        ax.set_yticks([])

                    ax.set_title(info_titles[0], fontsize=6 * size_factor)
                    ax_index += 1

                #####################################
                # Creates line plots               #
                ###################################

                if auto_bool:
                    divide_by = 500
                else:
                    divide_by = 1

                for data_ind in range(len(lineplot_data_list)):

                    # ax_index = raster_col_count+info_col_count+data_ind
                    ax = axs[ax_index]
                    color_list = ["green", "purple"]

                    pols_trimmed = pols[:2]

                    for pol_ind in range(len(pols_trimmed)):
                        lineplot_data = lineplot_data_list[data_ind][:, :, pol_ind]
                        for i in range(len(yticks)):
                            # print(y_index)
                            # y_index = yticks[i]
                            if lineplot_data_diff_bool[data_ind]:
                                data = (
                                    lineplot_data[yticks[i]]
                                    - lineplot_data[yticks[i - 1]]
                                ) / divide_by

                            else:
                                data = (lineplot_data[yticks[i]] / 10) / divide_by

                            if lineplot_data_diff_bool[data_ind] and i == 0:
                                pass
                            else:
                                if pol_ind == 0:
                                    ax.plot(
                                        np.zeros(data.shape) - 30 - yticks[i],
                                        color="gray",
                                        linewidth=0.2,
                                    )
                                cut_bad = 1 / (
                                    ~my_utils.coarse_band_flagging(
                                        Nfreqs=384,
                                        coarse_band_count=24,
                                        flag_edges=False,
                                    )
                                )
                                ax.plot(
                                    cut_bad * ((data - np.mean(data)) - 30 - yticks[i]),
                                    color=color_list[pol_ind],
                                    linewidth=0.3,
                                )

                        xticks_mod = len(data) / len(freq_array)
                        ax.set_xticks(xticks * xticks_mod)

                        ax.set_xlabel("freq (MHz)", fontsize=7 * size_factor)
                        ax.set_xticklabels(xticklabels, fontsize=5 * size_factor)

                        ax.set_ylim(-len(active_filled_obs_id_array), 0)
                        if lineplot_data_diff_bool[data_ind]:

                            ax.yaxis.tick_right()

                            ax.set_yticks(-np.array(yticks))
                            ax.set_yticklabels(
                                right_yticklabels, fontsize=10 * size_factor
                            )
                            ax.yaxis.set_label_position("right")
                            ax.set_ylabel("LST")
                        else:
                            ax.set_yticks([])
                        ax.set_yticks([])
                        # ax.set_yticks([])

                        ax.set_position(
                            [
                                col_width * ax_index + 0.2,
                                prev_ax_pos.y0,
                                col_width,
                                prev_ax_pos.height,
                            ]
                        )
                        ax.set_title(
                            f"{lineplot_data_titles[data_ind]}",
                            fontsize=6 * size_factor,
                        )
                    ax_index += 1

                if split_pointings:

                    ax.yaxis.tick_right()
                    ax.set_yticks(yticks)
                    ax.set_yticklabels(right_yticklabels, fontsize=3 * size_factor)
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel("LST")

                    """ax.yaxis.tick_right()
                    ax.set_yticks(-np.array(yticks))
                    ax.set_yticklabels(right_yticklabels,fontsize=3*size_factor)
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel('LST')
                    ax.set_position([col_width*(ax_index-1)+.2,prev_ax_pos.y0,col_width,prev_ax_pos.height])"""

                if ax_index == col_count - 1:

                    ax.yaxis.tick_right()

                    ax.set_yticks(-np.array(yticks))
                    ax.set_yticklabels(right_yticklabels, fontsize=10 * size_factor)
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel("LST")

                if split_pointings:
                    label = obs_list_ind
                else:
                    label = "all"

                fig.suptitle(
                    f"Date: {date_string}, {auto_tag} data({label})",
                    fontsize=5 * size_factor,
                    x=0.35,
                    y=prev_ax_pos.y1 + prev_ax_pos.y1 * 0.3 / height_factor,
                )

                ############################################################
                # Adds pointing changes and other metadata to figure      #
                ##########################################################

                ytickslocs = first_column_ax.get_yticks()
                xmin, _ = first_column_ax.get_xlim()

                ytickslocs = first_column_ax.transData.transform(
                    [(xmin, ytick_val) for ytick_val in ytickslocs]
                )

                size = fig.get_size_inches() * fig.dpi
                fig_width, fig_height = size

                for ind, loc in enumerate(ytickslocs):
                    if ind in pointing_change_results and ind != 0:

                        line_x, line_y = loc

                        fig.add_artist(
                            lines.Line2D(
                                [line_x / fig_width, 0.8],
                                [line_y / fig_height, line_y / fig_height],
                                color="magenta",
                            )
                        )
                        mini_text = (
                            f"RA:{round(ra_list[ind],2)},\n"
                            f"DEC:{round(dec_list[ind],2)}\n"
                            f"ALT:{round(alt_list[ind],2)},\n"
                            f"AZ:{round(az_list[ind],2)}"
                        )
                        plt.text(
                            0.9,
                            (line_y - 15) / fig_height,
                            mini_text,
                            transform=fig.transFigure,
                            fontsize=6 * size_factor,
                            verticalalignment="top",
                            bbox=dict(alpha=0.1, color="magenta"),
                        )

                pointing_number = my_utils.mwa_pointings(alt=alt_list[0], az=az_list[0])
                if not split_pointings:
                    textstr = "\n".join(
                        (
                            "Pointing center (1st obsid):",
                            f"    - PTING:{pointing_number}",
                            f"    - RA:{round(ra_list[0],2)}",
                            f"    - DEC:{round(dec_list[0],2)}\n",
                            f"    - ALT:{round(alt_list[0],2)}",
                            f"    - AZ:{round(az_list[0],2)}\n",
                            "Array config:",
                            "     -Phase II \n",
                            "Data source:",
                            "-    Nichole list",
                        )
                    )
                else:
                    textstr = "  ".join(
                        (
                            "Pting ctr (1st obsid):",
                            f"    PTING:{pointing_number}",
                            f"    RA:{round(ra_list[0],2)}",
                            f"    DEC:{round(dec_list[0],2)}",
                            f"    ALT:{round(alt_list[0],2)}",
                            f"    AZ:{round(az_list[0],2)}",
                        )
                    )

                props = dict(alpha=0.2)

                plt.text(
                    0.48,
                    (ytickslocs[0][1] + 100) / fig_height,
                    textstr,
                    transform=fig.transFigure,
                    fontsize=3 * size_factor,
                    verticalalignment="bottom",
                    bbox=props,
                )

                save_name = f"{date_string}_{auto_tag}_plot{label}.pdf"
                save_name = os.path.join(out_folder, save_name)
                print("saving to", save_name)
                fig.savefig(save_name, bbox_inches="tight")
                plt.close(fig)


# This part of the code is VERY UGLY. It is extremely spaghettified, hacky, insert your favorite pejoritive here,
# etc, and has needed a rewrite for a while. In essense, it reads in a set of h5 files,
# then combines them to make a big set of numpy arrays that can then be plotted. I hate it.
# However, it technically works.
"""
- file_list takes list of h5 files
- average_across tells the code which datasets it should average in time.
- ssins_data_folder can give the folder of seperate full-fledged SSINS files which can be used
  (useful if SSINS has already been run seperately e.g. on 40khz channel size data as opposed to 80khz data)
- metafits_folders only used to load in options files (for e.g. cutting bad times in data)
- load_options CAUTION: I have not been using this part of the code. I'm not sure it works as designed.
  This is part of the code that most needs a rewrite.
  Supposed to tell the code to load options.yml file if they are present in the metafits folder
- load_post_process_options same CAUTION statement as above.
  Supposed to tell the code to load options_post.yml files if they are present in the metafits folder.
- use_cross_array tells the function to use the mean_cross_array to make the stdv_array instead of reading it off.
  Useful if you need to alter which times are included in an observation
- add_ssins_mask adds the SSINS mask as one of the datasets to output
"""


def h5_file_collector(
    file_list,
    average_across=[],
    ssins_data_folder=None,
    metafits_folders=[""],
    load_options=False,
    load_post_process_options=False,
    use_cross_array=False,
    add_ssins_mask=True,
):
    if load_post_process_options:
        use_cross_array = True

    data_dict = {}

    # This chunk just makes sure the files are in obs_id order
    sorting_list = []
    for file in file_list:
        obs_id = (file.split("/")[-1]).split("_")
        sorting_list.append((obs_id, file))
    sorting_list.sort(key=lambda x: x[0])
    file_list = [entry[1] for entry in sorting_list]

    filled_time_array = np.array([])
    filled_obs_id_array = np.array([])
    filled_lst_array = np.array([])
    attr_dict = {}
    first_file_check = True
    for h5_name in file_list:
        obs_id = h5_name.split("/")[-1].split("_")[0]

        with h5py.File(h5_name, "r") as hf:
            if first_file_check:
                dataset_names = list(hf.keys())
                if "mean_cross_array" in dataset_names:
                    dataset_names.remove("mean_cross_array")

                for dataset_name in dataset_names:
                    data_dict[dataset_name] = None
                data_dict["time_mean"] = None
                if add_ssins_mask:
                    data_dict["SSINS_mask"] = None
                    data_dict["metric_ms_best_avail"] = None

            if average_across == [] and "blmean" in dataset_names:
                average_across = "blmean"
            elif average_across == [] and len(dataset_names) == 1:
                average_across = [dataset_names[0]]

            time_array_1d = hf[dataset_names[0]].attrs["time_array_1d"]

            lst_array = hf[dataset_names[0]].attrs["lst_array"]
            obs_id_array = np.full_like(time_array_1d, obs_id)

            if first_file_check:

                expected_time_diff = time_array_1d[1] - time_array_1d[0]

                for attr in hf[dataset_names[0]].attrs.keys():
                    attr_dict[attr] = hf[dataset_names[0]].attrs[attr]

                # Note while technically this metadata can vary between obs_id's,
                # this method should not be used if this is the case.
                # Also ignore the time array from this
            else:

                time_diff = time_array_1d[0] - prev_end_time
                index_gap = np.round(time_diff / expected_time_diff, 3) - 1

                time_array_1d = np.append(
                    np.full(int(index_gap), np.nan), time_array_1d, axis=0
                )
                obs_id_array = np.append(
                    np.full(int(index_gap), np.nan), obs_id_array, axis=0
                )
                lst_array = np.append(
                    np.full(int(index_gap), np.nan), lst_array, axis=0
                )

                correct_length = len(time_array_1d)

            filled_time_array = np.append(filled_time_array, time_array_1d, axis=0)
            filled_obs_id_array = np.append(filled_obs_id_array, obs_id_array, axis=0)
            filled_lst_array = np.append(filled_lst_array, lst_array, axis=0)

            prev_end_time = time_array_1d[-1]

            # print(dir(hf['blmean']))
            for dataset_name in dataset_names:

                if dataset_name == "stdv_array" and use_cross_array:
                    mean_cross_array = np.array(hf["mean_cross_array"])

                    if post_time_cuts is not None:

                        remove_times = np.array(
                            [
                                post_time_cuts[0] > t_ind or t_ind > post_time_cuts[1]
                                for t_ind in range(len(mean_cross_array))
                            ]
                        )
                    else:
                        remove_times = np.array(
                            [False for t_ind in range(len(mean_cross_array))]
                        )

                    data = vis_plt.EAVILS_variance(
                        mean_cross_array,
                        remove_times=remove_times,
                        output_stdv=True,
                        maintain_time_dimension=True,
                    )

                else:
                    dataset = hf[dataset_name]

                    data = np.array(dataset)

                if load_post_process_options:
                    post_options = {}
                    for metafits_folder in metafits_folders:
                        post_options.update(
                            vis_plt.options_load(
                                metafits_folder, int(obs_id), quiet_mode=True, post=True
                            )
                        )
                    if "time_cuts" in post_options:
                        post_time_cuts = list(post_options["time_cuts"])

                        if post_time_cuts[1] < 0:

                            post_time_cuts[1] = len(data) + post_time_cuts[1]
                    else:
                        post_time_cuts = None

                else:
                    post_time_cuts = None

                if post_time_cuts is not None:
                    # fmt: off
                    data = data[post_time_cuts[0]:post_time_cuts[1]]

                if dataset_name in average_across:
                    time_averaged_data = np.nanmean(data, axis=0)
                    time_averaged_data = time_averaged_data * np.full(
                        data.shape, 1
                    )  # giving it same shape as other arrays

                if not first_file_check:

                    nanshape = [correct_length - data.shape[0]] + list(data.shape[1:])
                    data = np.append(np.full(nanshape, np.nan), data, axis=0)

                    if dataset_name in average_across:
                        time_averaged_data = np.append(
                            np.full(nanshape, np.nan), time_averaged_data, axis=0
                        )

                if data_dict[dataset_name] is None:
                    data_dict[dataset_name] = data
                else:
                    data_dict[dataset_name] = np.append(
                        data_dict[dataset_name], data, axis=0
                    )

                if dataset_name in average_across:
                    if data_dict["time_mean"] is None:
                        data_dict["time_mean"] = time_averaged_data
                    else:
                        data_dict["time_mean"] = np.append(
                            data_dict["time_mean"], time_averaged_data, axis=0
                        )

                if add_ssins_mask and dataset_name == "metric_ms":
                    if ssins_data_folder is not None:
                        try:
                            h5_name = os.path.join(
                                ssins_data_folder, f"{obs_id}_SSINS_data.h5"
                            )
                            ins = INS(h5_name, run_check=False)
                            ins_array = copy.deepcopy(ins.metric_ms)

                            ins, _ = vis_plt.SSINS_mask_maker(ins)
                            SSINS_mask = ins.metric_array.mask

                            evens = np.arange(0, ins_array.shape[1], 2)
                            odds = evens + 1

                            ins_array = (
                                ins_array[:, evens, :] + ins_array[:, odds, :]
                            ) / 2
                            SSINS_mask = (
                                SSINS_mask[:, evens, :] | SSINS_mask[:, odds, :]
                            )

                        except OSError:
                            read_masks_directly_bool = True
                    else:
                        read_masks_directly_bool = True
                    if read_masks_directly_bool:
                        ins_array = np.array(hf["metric_ms"])
                        SSINS_mask = np.array(hf["mask"])

                    if load_options:
                        options = {}
                        for metafits_folder in metafits_folders:
                            options.update(
                                vis_plt.options_load(
                                    metafits_folder, int(obs_id), quiet_mode=True
                                )
                            )
                        if "time_cuts" in options:
                            time_cuts = options["time_cuts"]
                            # fmt: off
                            SSINS_mask = SSINS_mask[time_cuts[0]:time_cuts[1]]
                            ins_array = ins_array[time_cuts[0]:time_cuts[1]]
                    if load_post_process_options:
                        if post_time_cuts is not None:
                            # fmt: off
                            SSINS_mask = SSINS_mask[post_time_cuts[0]:post_time_cuts[1]]
                            ins_array = ins_array[post_time_cuts[0]:post_time_cuts[1]]

                    if data_dict["metric_ms_best_avail"] is None:
                        data_dict["metric_ms_best_avail"] = ins_array

                    else:
                        ins_nanshape = [correct_length - ins_array.shape[0]] + list(
                            ins_array.shape[1:]
                        )
                        ins_array = np.append(
                            np.full(ins_nanshape, np.nan), ins_array, axis=0
                        )
                        data_dict["metric_ms_best_avail"] = np.append(
                            data_dict["metric_ms_best_avail"], ins_array, axis=0
                        )

                    if data_dict["SSINS_mask"] is None:
                        data_dict["SSINS_mask"] = SSINS_mask

                    else:
                        mask_nanshape = [correct_length - SSINS_mask.shape[0]] + list(
                            SSINS_mask.shape[1:]
                        )
                        SSINS_mask = np.append(
                            np.full(mask_nanshape, np.nan), SSINS_mask, axis=0
                        )
                        data_dict["SSINS_mask"] = np.append(
                            data_dict["SSINS_mask"], SSINS_mask, axis=0
                        )

            first_file_check = False

    return (
        data_dict,
        filled_time_array,
        filled_obs_id_array,
        filled_lst_array,
        attr_dict,
    )


def obs_id_change_identifier(obs_id_array):
    last_item = 0
    yticks = []
    for ind, item in enumerate(obs_id_array):
        if item != last_item:
            if not np.isnan(item):
                yticks.append(ind)
        last_item = item
    return yticks


def raw_pointing_list(
    metafits_folders, obs_id_list, return_all=False, strictly_same=False
):
    """
    returns obs_ids where pointing changes occur and optionally some other things which can be
    used as inputs for pointing_change_identifier() below
    """
    if not isinstance(obs_id_list[0], str):
        print(
            "Warning: Correcting type of obs_id_list argument of raw_pointing_list to string"
        )
        obs_id_list = [str(obs_id) for obs_id in obs_id_list]
    coord_dict = {}
    for metafits_folder in metafits_folders:
        for file in os.listdir(metafits_folder):
            if ".metafits" in file:
                metafits = fits.open(os.path.join(metafits_folder, file))
                obs_id = file.split(".")[0]

                if obs_id in obs_id_list:
                    ra = metafits["primary"].header["RA"]
                    dec = metafits["primary"].header["DEC"]
                    alt = metafits["primary"].header["ALTITUDE"]
                    az = metafits["primary"].header["AZIMUTH"]
                    coord_dict[obs_id] = (ra, dec, alt, az)
    if strictly_same:
        missing_list = []
        for obs_id in obs_id_list:
            if obs_id not in coord_dict.keys():
                missing_list.append(obs_id)
        if len(missing_list) > 0:
            raise Exception(f"Missing the following obs_id: {missing_list}")

    ra_list = []
    dec_list = []
    alt_list = []
    az_list = []
    present_obs_id_list = []

    for obs_id in obs_id_list:

        if obs_id in list(coord_dict.keys()):

            ra, dec, alt, az = coord_dict[str(int(obs_id))]
            ra_list.append(ra)
            dec_list.append(dec)
            alt_list.append(alt)
            az_list.append(az)
            present_obs_id_list.append(obs_id)

    pointing_change_results = []
    pointing_change_obs_ids = []
    for ind, entry in enumerate(alt_list - np.roll(alt_list, 1)):
        if np.abs(entry) > 1e-2:

            pointing_change_obs_ids.append(present_obs_id_list[ind])

    for ind, obs_id in enumerate(obs_id_list):
        if obs_id in pointing_change_obs_ids:
            pointing_change_results.append(ind)
    if return_all:
        return (
            ra_list,
            dec_list,
            alt_list,
            az_list,
            pointing_change_results,
            pointing_change_obs_ids,
        )
    else:
        return pointing_change_obs_ids


def pointing_change_identifier(metafits_folders, filled_obs_id_array):
    """
    Returns a list of right ascensions, decs and pointing change indices relative to the array of unique obs_ids.
    Since this list of unique obs_ids is also how the yticks are built,
    it later allows us to place our pointing change lines.
    Also returns the time indices where they occur in the data array.
    """
    unique_obs_ids = np.unique(filled_obs_id_array)
    unique_obs_ids = [
        str(int(obs_id)) for obs_id in unique_obs_ids if not np.isnan(obs_id)
    ]

    (
        ra_list,
        dec_list,
        alt_list,
        az_list,
        pointing_change_results,
        pointing_change_obs_ids,
    ) = raw_pointing_list(metafits_folders, obs_id_list=unique_obs_ids, return_all=True)

    obs_id_indexer = {}
    changepoints = obs_id_change_identifier(filled_obs_id_array)

    for ind in range(len(unique_obs_ids)):
        obs_id_indexer[unique_obs_ids[ind]] = changepoints[ind]
    p_c_time_indices = []
    for p_c_obs_id in pointing_change_obs_ids:
        p_c_time_indices.append(obs_id_indexer[p_c_obs_id])
    p_c_time_indices.append(len(filled_obs_id_array) + 1)

    return (
        ra_list,
        dec_list,
        alt_list,
        az_list,
        pointing_change_results,
        p_c_time_indices,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process command line options.")
    # Positional argument for data_folders
    parser.add_argument(
        "data_folders", nargs="*", type=str, help="Paths to the data folders"
    )

    #
    parser.add_argument(
        "-m",
        "--metafits_folders",
        nargs="*",
        type=str,
        default=[""],
        help="Paths to the metafits folders",
        required=True,
    )
    parser.add_argument(
        "-l",
        "--list",
        type=str,
        default="",
        help="List of obs_ids to output",
        required=True,
    )
    parser.add_argument(
        "-r", "--receiver_data_usage", action="store_true", help="Receiver data usage"
    )
    parser.add_argument(
        "-b",
        "--bypass_full_array",
        action="store_true",
        help="Flag to bypass processing full array data",
    )
    parser.add_argument(
        "-s", "--split", action="store_true", help="Split output into different plots"
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        default=f"time_ext_out_{formatted_date}",
        help="output folder for generated plots",
    )

    args = parser.parse_args()

    main(
        data_folders=args.data_folders,
        metafits_folders=args.metafits_folders,
        receiver_data_usage=args.receiver_data_usage,
        bypass_full_array=args.bypass_full_array,
        obs_id_list_file=args.list,
        split_pointings=args.split,
        out_folder=args.output_folder,
    )
