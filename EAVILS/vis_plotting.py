#!/Users/elillesk/miniconda3/envs/my_env/bin/python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


import astropy.io.fits as fits

import scipy.stats as stats

import my_utils



import yaml
import os
import copy
import gc
import h5py
from collections import defaultdict


from SSINS import SS
from SSINS import INS
from SSINS.plot_lib import image_plot
from SSINS.data import DATA_PATH
from SSINS import MF
from SSINS import util


##############################################################################################
# Note! This function is not used for EAVILS at all, it's used to make "movies" of visibilities.


'''
import cv2
def uv_movie_maker(obs_id, uvfits_folder, output_path, uvd=None, autoscale=True):

    if not os.path.exists(output_path):
        os.makedirs(output_path, mode=0o777)

    obs_id = str(obs_id)
    if int(obs_id) < 1156551320:  # Checcks if is phase I or phase II observation
        extent = 2000  # Approximate extent of the array in meters
    else:
        extent = 500

    if type(uvd) is None:
        uvd, _ = my_utils.uvfits_reader(
            obs_id, uvfits_folder, conjugate_baselines=True, experimental_ant_check=True
        )

    else:
        print(
            "UVData object given in function argument, "
            "please ensure it has had the conjugate_bls method applied with v>0 for proper display"
        )

    freq_ranges = ["TV6", "TV7", "TV8", "TV9"]
    middle_indices = {}
    for tv in freq_ranges:
        middle_indices[tv] = int(
            np.average(
                np.arange(len(uvd.freq_array))[
                    my_utils.make_freq_mask(uvd.freq_array, tv)
                ]
            )
        )

    data = uvd.data_array.reshape(uvd.Ntimes, uvd.Nbls, uvd.Nfreqs, uvd.Npols)
    uvw = uvd.uvw_array.reshape(uvd.Ntimes, uvd.Nbls, 3)

    time_interval = round(
        ((np.unique(uvd.time_array)[1] - np.unique(uvd.time_array[0])) * 86400)[0], 2
    )

    # Loop through pols and tv bandS
    for pol in ["XX", "YY"]:
        for tv in freq_ranges:
            if pol == "XX":
                pol_ind = 0
            elif pol == "YY":
                pol_ind = 1

            f_ind = middle_indices[tv]
            freq = uvd.freq_array[f_ind]
            freq_string = str(round(freq / 10**6, 2)).replace(".", ",")

            # Images from previous runs to be cleaned up. Note this means if you want to save anything from a
            # previous run you should run this in a new folder
            clean_up_images = [
                img
                for img in os.listdir(output_path)
                if img.endswith(".png")
                and img.startswith(str(obs_id) + "_p" + pol + "_f" + freq_string)
            ]
            for image in clean_up_images:
                os.remove(os.path.join(output_path, image))

            horizontal_line = [
                np.arange(-extent, extent),
                np.zeros(2 * extent),
            ]  # plots lines representing directions in the uv plane
            vertical_line = [np.zeros(extent), np.arange(0, extent)]

            # loop through tickers and axes
            print(f"producing individual plots for {pol} freq {freq_string} MHz")
            datasets = []
            for t_ind1 in range(uvd.Ntimes - 1):
                t_ind2 = t_ind1 + 1
                datasets.append(
                    [
                        np.abs(data[t_ind1, :, f_ind, pol_ind]),
                        np.abs(
                            data[t_ind1, :, f_ind, pol_ind]
                            - data[t_ind2, :, f_ind, pol_ind]
                        ),
                        # This is a little confusing looking,
                        # but all it's doing is modifying the subtracted angles
                        # so insted of spanning from -2pi to 2pi,
                        # it wraps the values so they go  from -pi to pi.
                        # So e.g. 5/3 pi becomes -1/3 pi, -4/3 pi becomes 2/3 pi, etc
                        (
                            np.angle(data[t_ind1, :, f_ind, pol_ind])
                            - np.angle(data[t_ind2, :, f_ind, pol_ind]) % (2 * np.pi)
                            - np.pi
                        )
                        % (2 * np.pi)
                        - np.pi,
                        np.abs(data[t_ind1, :, f_ind, pol_ind])
                        - np.abs(data[t_ind2, :, f_ind, pol_ind]),
                    ]
                )

            # Should have shape of Ntimes-1, number of data types (e.g. diff of abs, abs of diff, etc),
            # number of baselines
            datasets = np.array(datasets)

            # Order should match that of associated dataset
            cmaps = ("viridis", "plasma", my_utils.cust_cmap("smoothbow"), "coolwarm")

            plot_types = (
                "scatter",
                "scatter",
                "scatter",
                "scatter",
                "hist",
                "hist",
                "hist",
                "hist",
            )
            plot_count = len(plot_types)

            data_indices = (
                0,
                1,
                2,
                3,
                0,
                1,
                2,
                3,
            )  # Picks out which data type in datasets we want to use

            titles = (
                "|V_a|",
                "|V_a - V_b|",
                "ang(V_a) - ang(V_b)",
                "|V_a|-|V_b|",
                "|V_a|",
                "|V_a - V_b|",
                "ang(V_a) - ang(V_b)",
                "|V_a|-|V_b|",
            )

            vlim_defaults = [
                (0, 10000),
                (0, 1000),
                (-np.pi, np.pi),
                (-500, 500),
                (),
                (),
                (),
                (),
            ]
            if autoscale:
                t_ind = 0
                vlims = vlim_defaults
                # Automatically sccales the vlims based on the initial state of the data

                data_index = 0
                values = datasets[t_ind, data_index, :]
                mean = np.mean(values)
                std_dev = np.std(values)
                lim = max(np.abs(mean + std_dev), np.abs(mean - std_dev))
                vlims[data_index] = (0, lim)

                data_index = 1
                values = datasets[t_ind, data_index, :]
                mean = np.mean(values)
                std_dev = np.std(values)
                lim = max(np.abs(mean + std_dev), np.abs(mean - std_dev))
                vlims[data_index] = (0, lim)

                data_index = 3
                values = datasets[t_ind, data_index, :]
                mean = np.mean(values)
                std_dev = np.std(values)
                lim = max(np.abs(mean + std_dev), np.abs(mean - std_dev))
                vlims[data_index] = (-lim, lim)

            else:
                vlims = vlim_defaults

            data_lims = [
                [np.min(datasets[:, data_index, :]), np.max(datasets[:, data_index, :])]
                for data_index in data_indices
            ]

            symmetric = (None, None, None, None, False, False, False, True)

            circular = (False, False, True, False, False, False, True, False)

            for t_ind1 in range(uvd.Ntimes - 1):
                t_ind2 = t_ind1 + 1
                row_count = 2
                col_count = len(data_indices) // 2

                fig, axs = plt.subplots(
                    nrows=row_count,
                    ncols=col_count,
                    figsize=(col_count * 4, row_count * 3),
                    dpi=300,
                )

                fig.suptitle(
                    "ObsId: "
                    + obs_id
                    + ", pol: "
                    + pol
                    + ", times: "
                    + str(round(time_interval * t_ind1, 1))
                    + "s - "
                    + str(round(time_interval * t_ind2, 1))
                    + "s, freq: "
                    + str(round(freq / 10**6, 2))
                    + " MHz"
                )
                fig.subplots_adjust(top=0.95)

                u = uvw[t_ind1, :, 0]
                v = uvw[t_ind1, :, 1]

                for i, ax in enumerate(axs.ravel()):
                    data_index = data_indices[i]
                    values = datasets[t_ind1, data_index, :]
                    if i < plot_count:
                        if plot_types[i] == "scatter":
                            # mag = np.abs(data[t_ind1,:,f_ind,0]-data[t_ind2,:,f_ind,0])
                            # mag_save.append(mag)
                            # ax.scatter(u,v,c=mag,alpha=1,s=.01,norm=colors.PowerNorm(3,vmin=0,vmax=1000),cmap='hot_r')

                            active_scatter = ax.scatter(
                                u,
                                v,
                                c=values,
                                alpha=0.25,
                                s=1,
                                vmin=vlims[i][0],
                                vmax=vlims[i][1],
                                cmap=cmaps[data_index],
                            )
                            # ax.scatter(u,v,c=mag,alpha=1,s=.01,vmin=0,vmax=2000)
                            ax.set_title(titles[i], fontsize=10)
                            # o = mpl.patches.Circle((0,0), 10, facecolor='black', edgecolor='black')
                            # ax.add_patch(o)
                            divider = make_axes_locatable(ax)
                            cax = divider.append_axes("right", size="5%", pad=0.05)
                            cbar = fig.colorbar(
                                active_scatter, cax=cax, orientation="vertical"
                            )
                            cbar.solids.set(alpha=1)
                            ax.plot(
                                horizontal_line[0],
                                horizontal_line[1],
                                color="black",
                                linewidth=0.5,
                                alpha=0.5,
                            )
                            ax.plot(
                                vertical_line[0],
                                vertical_line[1],
                                color="black",
                                linewidth=0.5,
                                alpha=0.5,
                            )
                            ax.text(
                                extent * (1 - 0.075),
                                extent * -0.025,
                                "%g" % (extent / 1000) + "km",
                                size=5,
                            )  # Placing distance label just below and to the left of right edge of distance axis

                            ax.get_xaxis().set_visible(False)
                            ax.get_yaxis().set_visible(False)

                            ax.tick_params(
                                axis="x",  # changes apply to the x-axis
                                which="both",  # both major and minor ticks are affected
                                bottom=False,  # ticks along the bottom edge are off
                                top=False,  # ticks along the top edge are off
                                labelbottom=False,
                            )

                            ax.tick_params(
                                axis="y",  # changes apply to the y-axis
                                which="both",  # both major and minor ticks are affected
                                bottom=False,  # ticks along the bottom edge are off
                                top=False,  # ticks along the top edge are off
                                labelbottom=False,
                            )
                            ax.axis("equal")

                        elif plot_types[i] == "hist":
                            ax.hist(values, bins=50)
                            ax.set_yscale("log")
                            if not symmetric[i]:
                                ax.set_xlim(data_lims[i])
                            else:
                                max_bound = np.max(np.abs(data_lims[i]))
                                ax.set_xlim([-max_bound, max_bound])
                            ax.set_ylim([0.5, uvd.Nbls / 2])

                            # Calculate mean, median, and standard deviation
                            if not circular[i]:
                                mean = np.mean(values)
                                median = np.median(values)
                                std_dev = np.std(values)

                            else:
                                mean = stats.circmean(values, low=-np.pi, high=np.pi)
                                std_dev = stats.circstd(values, low=-np.pi, high=np.pi)

                            # Add lines for mean, median, and standard deviation
                            ax.axvline(
                                mean,
                                color="r",
                                linestyle="dashed",
                                linewidth=2,
                                label=f"Mean: {mean:.2f}",
                            )

                            if not circular[i]:
                                ax.axvline(
                                    median,
                                    color="g",
                                    linestyle="dashed",
                                    linewidth=2,
                                    label=f"Median: {median:.2f}",
                                )

                            ax.axvline(
                                mean - std_dev,
                                color="orange",
                                linestyle="dotted",
                                linewidth=2,
                                label=f"Std Dev: {std_dev:.2f}",
                            )
                            ax.axvline(
                                mean + std_dev,
                                color="orange",
                                linestyle="dotted",
                                linewidth=2,
                            )
                            ax.legend(fontsize="x-small", markerscale=0.25)

                save_name = os.path.join(
                    output_path,
                    str(obs_id)
                    + "_p"
                    + pol
                    + "_f"
                    + freq_string
                    + "_t"
                    + str(t_ind1)
                    + ".png",
                )
                fig.savefig(save_name)

                plt.close(fig)

            print("sorting images")

            video_name = os.path.join(
                output_path, str(obs_id) + "_" + pol + "_" + freq_string + "_video.avi"
            )

            images = [
                img
                for img in os.listdir(output_path)
                if img.endswith(".png")
                and img.startswith(str(obs_id) + "_p" + pol + "_f" + freq_string)
            ]

            image_index = [
                (int(images[i].split("_t")[1].split(".")[0]), i)
                for i, img in enumerate(images)
            ]
            image_index = sorted(image_index, key=tuple[1])
            images = [images[image_index[i][1]] for i in range(len(images))]

            print("generating movie")
            frame = cv2.imread(os.path.join(output_path, images[0]))
            height, width, layers = frame.shape

            video = cv2.VideoWriter(video_name, 0, 1, (width, height))

            for image in images:
                video.write(cv2.imread(os.path.join(output_path, image)))

            cv2.destroyAllWindows()
            video.release()'''


#############################################################################################


"""
This function runs EAVILS and SSINS, as well as extracting some data about
receivers which are probably less useful (Which can be turned off via the
per_receiver=False argument).

The uvd_cross and uvd_autos options are in fact best used by giving an undiffed
SSINS ss object as input, rather than an actual uvd object, since an undiffed ss
object has all the attributes of a uvd object and some others.
That is how it is used in this module.

old_spectra_type is not useful, just left around to keep track of how to construct some statistics.
The options dict is meant to allow for obs_id specific options (for example cutting bad times).
"""


def spectra_maker(
    obs_id,
    uvfits_folder,
    output_path,
    uvd_cross=None,
    uvd_autos=None,
    add_SSINS=True,
    old_spectra_type=False,
    options={},
    per_receiver=True,
):
    obs_id = str(obs_id)
    mwax_switch_obs_id = 1318021528
    if int(obs_id) < mwax_switch_obs_id:
        print(f"obsid is less than {mwax_switch_obs_id}, assuming legacy correlator")
        mwax_bool = False
    else:
        print(f"date is greater than {mwax_switch_obs_id}, assuming MWAX correlator")
        mwax_bool = True

    if not os.path.exists(output_path):
        os.makedirs(output_path, mode=0o777)

    spectra_path = os.path.join(output_path, "output_spectra")
    if not os.path.exists(spectra_path):
        os.makedirs(spectra_path, mode=0o777)

    # Simply checks if uvd_cross was given as input,
    # if not then reads it and uvd_autos in from a file in the given folder
    if uvd_cross is None:
        uvd_cross, uvd_autos = my_utils.uvfits_reader(
            obs_id,
            uvfits_folder,
            split_autos=True,
            experimental_ant_check=True,
            use_ss_as_uvd=True,
        )

    print("Preparing data")

    if "time_cuts" in options.keys():
        time_cuts = options["time_cuts"]
        print(f"time_cuts set to {time_cuts}")
    else:
        time_cuts = None

    if "save_cross_receiver_data" in options.keys():
        save_cross_receiver_data = True
        print(f"save_cross_receiver_data set to {save_cross_receiver_data}")
    else:
        save_cross_receiver_data = False

    # This section just sets up proper tags and settings for auto and cross spectra
    bl_type_tag_list = []
    if uvd_autos is not None:
        bl_type_tag_list.append("auto")

    if uvd_cross is not None:
        bl_type_tag_list.append("cross")

    for bl_type_tag in bl_type_tag_list:
        if bl_type_tag == "auto":
            auto_bool = True
            uvd = uvd_autos
            del uvd_autos

        elif bl_type_tag == "cross":
            auto_bool = False
            uvd = uvd_cross
            del uvd_cross

        if time_cuts is not None:
            # fmt: off
            uvd.select(times=np.unique(uvd.time_array)[time_cuts[0]:time_cuts[1]])

        if add_SSINS:
            # Since ss is already loaded in (using an unsubtracted ss in place of a uvd object)
            # we can input the ss directly into SSINS
            # Returns a dict of INS's because function is built to be able to split into auto
            # and cross spectras if need be, but that's already been done so we immediately extract
            # the ins from the dict (in this case our dict has only one object).
            ins_dict = make_ssins_wrapper(
                ss=uvd,
                return_ins_only=True,
                spectrum_types=[bl_type_tag],
                mask_bool=False,
                time_cuts=time_cuts,
                flag_centers=not mwax_bool,
            )
            ins = ins_dict[bl_type_tag]

        # Sometimes the freq_array object includes surrounding brackets, deals with this
        if len(uvd.freq_array.shape) > 1:
            used_freq_array = uvd.freq_array[0]
        else:
            used_freq_array = uvd.freq_array

        pols = ["XX", "YY", "XY", "YX"]

        # IGNORE THIS
        #################################################################
        # This code isn't used anymore, but is useful to keep around for remembering how the
        # circular mean and stdv are calculated. No guarantee of reliability
        if old_spectra_type:
            # data = uvd.data_array.reshape(uvd.Ntimes,uvd.Nbls,uvd.Nfreqs,uvd.Npols)
            data_copy1 = np.zeros(
                (uvd.Ntimes + 1, uvd.Nbls, uvd.Nfreqs, uvd.Npols), dtype=complex
            )
            data_copy2 = np.zeros(
                (uvd.Ntimes + 1, uvd.Nbls, uvd.Nfreqs, uvd.Npols), dtype=complex
            )

            # fmt: off
            data_copy1[0:uvd.Ntimes, :, :, :] = data(uvd)
            data_copy2[1:uvd.Ntimes + 1, :, :, :] = data(uvd)

            print("Performing sky subtraction of phase angles")
            subtracted_data = np.angle(data_copy1) - np.angle(data_copy2)
            # This is a little confusing looking,
            # but all it's doing is modifying the subtracted angles so insted of
            # spanning from -2pi to 2pi,
            # it wraps the values so they go  from -pi to pi.
            # So e.g. 5/3 pi becomes -1/3 pi, -4/3 pi becomes 2/3 pi, etc
            subtracted_data = (subtracted_data[1:-1, :, :, :] % (2 * np.pi) - np.pi) % (
                2 * np.pi
            ) - np.pi
            print("Analyzing mean and standard deviation")

            circ_mean_array = stats.circmean(
                subtracted_data, axis=1, low=-np.pi, high=np.pi
            )
            circ_stdv_array = stats.circstd(
                subtracted_data, axis=1, low=-np.pi, high=np.pi
            )

            print("Summing visibilities")
            summed_amps = np.sum(np.abs(data), axis=1)
            mean_sub_summed_amps = summed_amps - np.average(summed_amps, axis=0)
            norm_mean_sub_summed_amps = mean_sub_summed_amps / np.average(
                summed_amps, axis=0
            )

            datasets = [
                circ_mean_array,
                circ_stdv_array,
                summed_amps,
                mean_sub_summed_amps,
                norm_mean_sub_summed_amps,
            ]
            titles = [
                "Circular mean of phase difference",
                "Circular stdev of phase difference",
                "sum(abs(vis))",
                "sum(abs(vis))-avg per freq",
                "(sum(abs(vis))-avg)/avg",
            ]
            cmaps = [
                my_utils.cust_cmap("smoothbow"),
                "Spectral",
                "viridis",
                "coolwarm",
                "coolwarm",
            ]
            vlims = [(-np.pi, np.pi), None, None, None, None]
            change_colorscale = ["log", None, None, None, None]
        ###########################################################################
        # STOP IGNORING

        # Only ever uses this part
        else:
            # This is where the EAVILS information is extracted.
            # Includes intermediate steps as well as final z_score so we can
            # plot intermediate steps and reconstruct the z_scores if necessary
            blmean_data, blmean_data_sub, stdv_array, z_score, mean_cross_array = (
                EAVILS(uvd, compute_cross=True)
            )

            h5_path = os.path.join(output_path, "h5_files")

            #########################################################
            # Constructs the SSINS mask
            if add_SSINS and not auto_bool:
                ins_copy, occ_dict = SSINS_mask_maker(ins, flag_centers=not mwax_bool)

                channel_width = f"{int(uvd.channel_width[0]/1000)}khz"
                occupancy_yaml = os.path.join(
                    output_path, f"{channel_width}_occupancy.yml"
                )

                if os.path.exists(occupancy_yaml):
                    with open(occupancy_yaml, "r") as yamlfile:
                        cur_yaml = yaml.safe_load(yamlfile)  # Note the safe_load
                else:
                    cur_yaml = {}

                cur_yaml[str(obs_id)] = {"SSINS": occ_dict}
                with open(occupancy_yaml, "w") as yamlfile:
                    yaml.safe_dump(cur_yaml, yamlfile)  # Also note the safe_dump

                mask_h5_prefix = os.path.join(h5_path, f"{str(obs_id)}_{channel_width}")
                ins_copy.write(mask_h5_prefix, output_type="mask", clobber=True)
            ######################################################################

            # Saves important data to an h5 file.
            if not os.path.exists(h5_path):
                os.makedirs(h5_path, mode=0o777)

            h5_name = os.path.join(
                h5_path, f"{str(obs_id)}_spectra_data_{bl_type_tag}.h5"
            )
            with h5py.File(h5_name, "w") as hf:

                save_data = [
                    ("blmean", blmean_data),
                    ("stdv_array", stdv_array),
                    ("mean_cross_array", mean_cross_array),
                ]
                if add_SSINS:
                    save_data.append(("metric_ms", ins.metric_ms))
                    if not auto_bool:
                        save_data.append(("mask", ins_copy.metric_array.mask))

                for name, array in save_data:
                    dset = hf.create_dataset(name, data=array)
                    dset.attrs["Ntimes"] = array.shape[0]
                    dset.attrs["Nfreqs"] = uvd.Nfreqs
                    dset.attrs["Nbls"] = uvd.Nbls
                    dset.attrs["Npols"] = uvd.Npols

                    dset.attrs["freq_array"] = used_freq_array
                    dset.attrs["time_array_1d"] = np.unique(uvd.time_array)
                    dset.attrs["lst_array"] = np.unique(uvd.lst_array)

            # Receiver stuff
            #####################################################################
            # This section deals with saving out some receiver dependent effects.
            # It's otherwise probably not useful, and nothing else relies on it.
            # It can be ignored unless you're specifically working on something
            # related to the receivers.
            if per_receiver:
                metafits_path = os.path.join(uvfits_folder, obs_id + ".metafits")
                rec_mask_dict, ant_names_included_dict = receiver_identify(
                    metafits_location=metafits_path, uvd=uvd, return_names=True
                )

                if auto_bool or save_cross_receiver_data:

                    receiver_h5_name = os.path.join(
                        h5_path, f"{str(obs_id)}_receiver_data_{bl_type_tag}.h5"
                    )

                    utf8_type = h5py.string_dtype("utf-8", 30)
                    with h5py.File(receiver_h5_name, "w") as hf:
                        for rec_choice in rec_mask_dict.keys():

                            dset = hf.create_dataset(
                                "receiver" + str(rec_choice),
                                data=((data(uvd))[:, rec_mask_dict[rec_choice], :, :]),
                            )
                            dset.attrs["freq_array"] = used_freq_array
                            dset.attrs["time_array_1d"] = np.unique(uvd.time_array)
                            dset.attrs["lst_array"] = np.unique(uvd.lst_array)
                            dset.attrs["ant_labels"] = np.array(
                                ant_names_included_dict[rec_choice], dtype=utf8_type
                            )
                            dset.attrs["Ntimes"] = uvd.Ntimes
                            dset.attrs["Nfreqs"] = uvd.Nfreqs
                            dset.attrs["Npols"] = uvd.Npols
                            dset.attrs["Nbls"] = sum(rec_mask_dict[rec_choice])

                        if save_cross_receiver_data:
                            all_same_rec_mask = rec_mask_dict[
                                list(rec_mask_dict.keys())[0]
                            ]
                            for rec_choice in rec_mask_dict.keys():
                                all_same_rec_mask = np.logical_or(
                                    all_same_rec_mask, rec_mask_dict[rec_choice]
                                )

                            dset = hf.create_dataset(
                                "different_receiver",
                                data=((data(uvd))[:, ~all_same_rec_mask, :, :]),
                            )
                            dset.attrs["freq_array"] = used_freq_array
                            dset.attrs["time_array_1d"] = np.unique(uvd.time_array)
                            dset.attrs["lst_array"] = np.unique(uvd.lst_array)
                            dset.attrs["Ntimes"] = uvd.Ntimes
                            dset.attrs["Nfreqs"] = uvd.Nfreqs
                            dset.attrs["Npols"] = uvd.Npols
                            dset.attrs["Nbls"] = sum(~all_same_rec_mask)

                ##############################################################
                pols = ["XX", "YY", "XY", "YX"]
                row_count = len(rec_mask_dict.keys())
                col_count = len(pols)

                fig, axs = plt.subplots(
                    nrows=row_count,
                    ncols=col_count,
                    figsize=(col_count * 4, row_count * 2),
                    dpi=300,
                )

                fig.suptitle(
                    "ObsId: " + str(obs_id) + ", <|V|;bl> - <<|V|;bl>;t> per receiver"
                )
                fig.subplots_adjust(top=0.95)

                xticks = np.arange(0, len(uvd.freq_array), 50)
                xticklabels = [
                    "%.1f" % (used_freq_array[tick] * 10 ** (-6)) for tick in xticks
                ]

                # time_interval = np.round(((np.unique(uvd.time_array)[1]-np.unique(uvd.time_array[0]))*86400),2)

                yticks = np.arange(0, uvd.Ntimes, 5)
                # time_names = np.arange(0, uvd.Ntimes*time_interval,time_interval)
                time_names = np.round(
                    (np.unique(uvd.time_array) - np.unique(uvd.time_array)[0]) * 86400
                )

                yticklabels = ["%.1f" % (time_names[tick]) for tick in yticks]
                for pol_ind in range(len(pols)):
                    for ind, rec in enumerate(rec_mask_dict.keys()):
                        receiver_mask = rec_mask_dict[rec]
                        ax = axs[ind, pol_ind]

                        active_plot = ax.imshow(
                            np.mean(
                                np.abs(
                                    data(uvd)[:, :, :, pol_ind][:, receiver_mask, :]
                                ),
                                axis=1,
                            )
                            - np.mean(
                                np.mean(
                                    np.abs(
                                        data(uvd)[:, :, :, pol_ind][:, receiver_mask, :]
                                    ),
                                    axis=1,
                                ),
                                axis=0,
                            ),
                            aspect=3,
                            cmap="coolwarm",
                        )
                        fig.colorbar(active_plot, orientation="vertical")
                        ax.set_xticks(xticks)
                        ax.set_xticklabels(xticklabels, fontsize=6)
                        ax.set_yticks(yticks)
                        ax.set_yticklabels(yticklabels, fontsize=6)
                        ax.set_title(f"{pols[pol_ind]}, receiver: {rec}", fontsize=8)
                save_name = os.path.join(
                    spectra_path, f"{str(obs_id)}_per_receiver_{bl_type_tag}.pdf"
                )
                print(f"saving {save_name}")
                fig.savefig(save_name)
                plt.close("all")
            ################################################################
            # End of receiver stuff

            # Plotting section!
            ###########################################################################
            # For each section of the plot, there are a few things we need,
            # which we save as lists to allow some modularity.
            # The datasets deals with the actual data we want to plot for each section
            datasets = [
                blmean_data,
                blmean_data_sub,
                z_score,
                stdv_array / np.sqrt(uvd.Nbls),
                z_score,
            ]
            # Titles just gives titles to each section
            titles = [
                "<|V|;bl>",
                "<|V|;bl> - <<|V|;bl>;t>",
                "z-score",
                "z-score divisor",
                "z-score histogram",
            ]
            # cmaps gives color schemes for relevant plots, otherwise just leaves blank as ''
            cmaps = ["viridis", "coolwarm", "coolwarm", "", ""]
            # Vlims deals with defined limits for plot scale when desired
            vlims = [None, None, (-5, 5), None, None]
            # This is just the plot types. ALlowed options for now are 'im' for image plot,
            # 'line' for a simple line plot, and 'hist' for histogram plots
            plot_types = ["im", "im", "im", "line", "hist"]

        # Simply adds the SSINS plots onto the ends of the plotting lists
        if add_SSINS:
            titles.append("SSINS")
            datasets.append(ins.metric_ms)
            cmaps.append("coolwarm")
            vlims.append((-5, 5))
            plot_types.append("im")

        # Builds figure
        row_count = len(datasets)
        col_count = len(pols)

        fig, axs = plt.subplots(
            nrows=row_count,
            ncols=col_count,
            figsize=(col_count * 4, row_count * 2),
            dpi=300,
        )

        fig.suptitle("ObsId: " + str(obs_id) + ", " + bl_type_tag)
        fig.subplots_adjust(top=0.95)
        fig.tight_layout()

        for pol_ind in range(len(pols)):
            for ind in range(row_count):
                ax = axs[ind, pol_ind]
                data_index = ind
                values = datasets[data_index][:, :, pol_ind]

                if plot_types[ind] == "im":
                    default_aspect = len(values[0]) / len(values)
                    desired_aspect = 0.5

                    if vlims[ind] is None:
                        active_plot = ax.imshow(
                            values,
                            aspect=default_aspect * desired_aspect,
                            cmap=cmaps[ind],
                        )
                    else:

                        active_plot = ax.imshow(
                            values,
                            aspect=default_aspect * desired_aspect,
                            cmap=cmaps[ind],
                            vmin=vlims[ind][0],
                            vmax=vlims[ind][1],
                        )

                    fig.colorbar(active_plot, orientation="vertical")

                    xticks = np.arange(12, len(used_freq_array), 50)
                    xticklabels = [
                        "%.0f" % (used_freq_array[tick] * 10 ** (-6)) for tick in xticks
                    ]
                    # time_interval = uvd.time
                    # time_interval = 2
                    yticks = np.arange(0, len(values), 5)
                    # time_names = np.arange(0, len(values)*time_interval,time_interval)
                    time_names = np.round(
                        (np.unique(uvd.time_array) - np.unique(uvd.time_array)[0])
                        * 86400
                    )

                    yticklabels = ["%.1f" % (time_names[tick]) for tick in yticks]

                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticklabels, fontsize=6)
                    ax.set_yticks(yticks)
                    ax.set_yticklabels(yticklabels, fontsize=6)
                elif plot_types[ind] == "hist":

                    try:

                        freq_mask = ~my_utils.coarse_band_flagging(
                            Nfreqs=uvd.Nfreqs,
                            coarse_band_count=24,
                            flag_centers=not mwax_bool,
                        )
                        ax.hist(
                            values[:, freq_mask].flatten(),
                            bins=100,
                            histtype="step",
                            density=True,
                        )

                        ax.set_yscale("log")

                        m = np.arange(-4, 4.1, 0.1)
                        ax.plot(m, stats.norm.pdf(m))

                        max_val = np.round(np.max(values[:, freq_mask].flatten()))
                        interval = 2 * max_val / 8
                        xticks = np.arange(-max_val, max_val + interval, interval)
                        xticklabels = xticks
                        ax.set_xticks(xticks)
                        ax.set_xticklabels(xticklabels, fontsize=6)
                    except ValueError as error:
                        print(
                            f"A ValueError occured plotting the histogram for {bl_type_tag} baselines "
                            f"in {pols[pol_ind]} pol, skipping:",
                            error,
                        )

                elif plot_types[ind] == "line":
                    ax.plot(values[0])
                    my_utils.forceAspect(ax, aspect=0.5)
                    """xticks = np.arange(0, len(used_freq_array), 50)
                    xticklabels = ['%.1f' % (used_freq_array[tick]* 10 ** (-6)) for tick in xticks]"""
                    xticks = np.arange(12, len(used_freq_array), 50)
                    xticklabels = [
                        "%.0f" % (used_freq_array[tick] * 10 ** (-6)) for tick in xticks
                    ]

                    ax.set_xticks(xticks)
                    ax.set_xticklabels(xticklabels, fontsize=6)
                    ax.set_xlim(0, len(values[0]))

                    box = ax.get_position()
                    # box.x0 = box.x0 - .021
                    # box.x1 = box.x1 - .021
                    adjust_x = -0.021
                    box.x0 = box.x0 + adjust_x
                    box.x1 = box.x1 + adjust_x
                    ax.set_position(box)

                ax.set_title(f"{pols[pol_ind]}, {titles[ind]}", fontsize=8)

        save_name = os.path.join(
            spectra_path, f"{str(obs_id)}_output_spectra_{bl_type_tag}.pdf"
        )

        print(f"saving {save_name}")
        fig.savefig(save_name)
        plt.close("all")

    local_dict = locals()
    to_delete = [
        name for name, val in local_dict.items() if isinstance(val, np.ndarray)
    ]

    for name in to_delete:
        del local_dict[name]

    # Force garbage collection to free memory
    gc.collect()
    print("")
    print("")


def data(uvd):
    return uvd.data_array.reshape(uvd.Ntimes, uvd.Nbls, uvd.Nfreqs, uvd.Npols)


def receiver_identify(metafits_location, uvd, return_names=False):
    metafits = fits.open(metafits_location)
    tilenames = metafits["TILEDATA"].data.field("TileName")
    receivers = metafits["TILEDATA"].data.field("Rx")
    name_to_rec_dict = {}
    for ind in range(len(tilenames)):
        tilename = tilenames[ind]
        receiver = receivers[ind]
        name_to_rec_dict[tilename] = receiver

    antenna_names_fix = [
        ant_name.rstrip() for ant_name in uvd.antenna_names
    ]  # Gets rid of unnecessary whitespace

    ant_num_name_dict = {}
    for i, name in enumerate(antenna_names_fix):
        ant_num = uvd.antenna_numbers[i]
        ant_num_name_dict[ant_num] = name
    receiver_mask = np.zeros(uvd.Nbls, dtype=bool)

    ant_1_array_1d = uvd.ant_1_array.reshape(uvd.Ntimes, uvd.Nbls)[0]
    ant_2_array_1d = uvd.ant_2_array.reshape(uvd.Ntimes, uvd.Nbls)[0]

    ant_names_included_dict = defaultdict(list)

    rec_mask_dict = {}
    for rec_choice in np.unique(receivers):
        receiver_mask = np.zeros(uvd.Nbls, dtype=bool)

        for i in range(len(ant_1_array_1d)):
            ant1 = ant_1_array_1d[i]
            ant2 = ant_2_array_1d[i]
            name1 = ant_num_name_dict[ant1]
            name2 = ant_num_name_dict[ant2]
            rec1 = name_to_rec_dict[name1]
            rec2 = name_to_rec_dict[name2]
            if rec_choice == rec1 and rec_choice == rec2:
                ant_names_included_dict[rec_choice].append(name1 + "_" + name2)
                receiver_mask[i] = True

        rec_mask_dict[rec_choice] = receiver_mask

    if return_names:
        return rec_mask_dict, ant_names_included_dict
    else:
        return rec_mask_dict


# WIP
# Deals with making the SSINS from a variety of input choices
def make_ssins_wrapper(
    obs_id=None,
    uvfits_folder="",
    ss=None,
    output_path=None,
    return_ins_only=False,
    spectrum_types=["cross"],
    mask_bool=True,
    time_cuts=None,
    override_time_cuts=False,
    assume_already_diffed=False,
    flag_centers=True,
):
    # Either an obs_id and uvfits folder must be provided or an ss object must be provided
    print(spectrum_types)
    load_from_file_bool = False

    if ss is None:
        if obs_id is None or uvfits_folder == "":
            raise Exception("Must include obs_id and uvfits_folder if ss is empty")
        else:
            load_from_file_bool = True

    if load_from_file_bool:
        if not isinstance(obs_id, str):
            obs_id = str(obs_id)
        ss = SS()
        full_uvfits_path = os.path.join(uvfits_folder, obs_id + ".uvfits")

        ss.read(full_uvfits_path, diff=True)

    else:
        ss = copy.deepcopy(ss)
        if not assume_already_diffed:
            ss.diff()

    if load_from_file_bool or override_time_cuts:
        if time_cuts is not None:
            # fmt: off
            ss.select(times=np.unique(ss.time_array)[time_cuts[0]:time_cuts[1]])

    if mask_bool:
        custom = np.zeros(ss.data_array.shape, dtype=bool)
        for ind, freq_channel in enumerate(
            my_utils.coarse_band_flagging(
                Nfreqs=ss.Nfreqs, coarse_band_count=24, flag_centers=flag_centers
            )
        ):
            if freq_channel:
                custom[:, :, ind, :] = True
        ss.apply_flags(flag_choice="custom", custom=custom)

    ins_dict = {}

    for spectrum_type in spectrum_types:
        ins = INS(ss, spectrum_type=spectrum_type)
        ins_dict[spectrum_type] = ins

    if return_ins_only:
        return ins_dict

    # IGNORE THE REMAINDER
    # Isn't really used unless making seperate SSINS plots
    fig = plt.figure(figsize=(12, 7), dpi=300)
    ax_SSINS_XX = fig.add_subplot()

    make_ssins_plot(
        fig, ax_SSINS_XX, ins, "XX", title=str(obs_id) + " SSINS: XX z-scores"
    )
    out_folder = os.path.join(output_path, str(obs_id) + "_XX_ssins.pdf")
    fig.savefig(out_folder)

    fig = plt.figure(figsize=(12, 7), dpi=300)
    ax_SSINS_YY = fig.add_subplot()

    make_ssins_plot(
        fig, ax_SSINS_YY, ins, "YY", title=str(obs_id) + " SSINS: YY z-scores"
    )
    out_folder = os.path.join(output_path, str(obs_id) + "_YY_ssins.pdf")
    fig.savefig(out_folder)


def SSINS_mask_maker(ins, flag_centers=True):
    ins_copy = copy.deepcopy(ins)
    ins_copy.metric_array.mask = np.logical_or(
        ins_copy.metric_array.mask,
        my_utils.coarse_band_flagging(
            Nfreqs=ins_copy.Nfreqs,
            Ntimes=ins_copy.Ntimes,
            Npols=ins_copy.Npols,
            coarse_band_count=24,
            flag_centers=flag_centers,
        ),
    )

    init_flag_occ = np.sum(ins_copy.metric_array.mask)
    shape_dict = {
        "TV6": [1.74e8, 1.81e8],
        "TV7": [1.81e8, 1.88e8],
        "TV8": [1.88e8, 1.95e8],
        "TV9": [1.95e8, 2.02e8],
    }
    sig_thresh = {shape: 5 for shape in shape_dict}
    sig_thresh["narrow"] = 5
    sig_thresh["streak"] = 10
    mf = MF(
        ins_copy.freq_array,
        sig_thresh,
        shape_dict=shape_dict,
        tb_aggro=0.4,
        broadcast_streak=True,
    )
    mf.apply_match_test(
        ins_copy, event_record=True, time_broadcast=True, freq_broadcast=False
    )

    occ_dict = util.calc_occ(ins_copy, mf, init_flag_occ, lump_narrowband=True)
    return ins_copy, occ_dict


# IGNORE THIS
# Only necessary for saving out seperate SSINS plots
def make_ssins_plot(fig, ax, ins, pol, title=""):
    if pol == "XX" or pol == "xx":
        pol = "XX"
        pol_ind = 0
    elif pol == "YY" or pol == "yy":
        pol = "YY"
        pol_ind = 1
    elif pol == "XY" or pol == "xy":
        pol = "XY"
        pol_ind = 2
    elif pol == "YX" or pol == "yx":
        pol = "YX"
        pol_ind = 3
    else:
        print("polarization given:", pol)
        print("Not supported!")
        return -1
    if title == "":
        title = pol + " z-scores"

    xticks = np.arange(12, len(ins.freq_array), 50)
    xticklabels = ["%.0f" % (ins.freq_array[tick] * 10 ** (-6)) for tick in xticks]

    time_interval = round((ins.time_array[1] - ins.time_array[0]) * 86400, 3)
    yticks = np.arange(0, len(ins.time_array), 5)
    time_names = np.arange(0, len(ins.time_array) * time_interval, time_interval)
    yticklabels = ["%.1f" % (time_names[tick]) for tick in yticks]

    # The z-scores are stored in the metric_ms parameter.

    image_plot(
        fig,
        ax,
        ins.metric_ms[:, :, pol_ind],
        title=title,
        xticks=xticks,
        xlabel="Freq (MHz)",
        xticklabels=xticklabels,
        yticks=yticks,
        yticklabels=yticklabels,
        ylabel="Time (sec)",
        cmap="coolwarm",
        midpoint=True,
        mask_color="black",
        vmin=-4,
        vmax=4,
    )


def EAVILS(uvd_or_data, compute_cross=False):
    """
    Expected Amplitude of VisibILities Spectra
    Expects data with the format of a numpy array with the following axes order: (bltime, freq, pol)
    Can be given either a uvd object (or undiffed ss object) or just the data array from one as an argument
    compute_cross allows for the evaluation of what I have called the mean_cross_array (not the best nomenclature)
    that allows the variance to be calculated when times are removed.
    """
    if isinstance(uvd_or_data, np.ndarray):
        pass
    else:
        """Ntimes = uvd_or_data.Ntimes
        Nbls = uvd_or_data.Nbls
        Nfreqs = uvd_or_data.Nfreqs
        Npols = uvd_or_data.Npols"""
        uvd_or_data = data(uvd_or_data)

    Ntimes, Nbls, Nfreqs, Npols = uvd_or_data.shape

    blmean_data = np.mean(np.abs(uvd_or_data), axis=1)

    blmean_data_sub = blmean_data - np.mean(blmean_data, axis=0)

    stdv_array = np.sqrt(np.mean(np.var(np.abs(uvd_or_data), axis=0, ddof=1), axis=0))

    stdv_array = stdv_array * np.full(
        (Ntimes, Nfreqs, Npols), 1
    )  # giving it same shape as other arrays
    z_score = np.sqrt(Nbls) * blmean_data_sub / stdv_array
    if compute_cross:
        mean_cross_array = (
            np.einsum("ijkl,pjkl->ipkl", np.abs(uvd_or_data), np.abs(uvd_or_data))
            / Nbls
        )

    else:
        mean_cross_array = None

    # Note: the following will compute mean variance (for freq index and pol index of 0) as well:
    # (np.sum(np.diag(mean_cross_array[:,:,0,0]))/uvd.Ntimes - np.sum(mean_cross_array[:,:,0,0])/uvd.Ntimes**2)*
    # (uvd.Ntimes/(uvd.Ntimes-1))

    return blmean_data, blmean_data_sub, stdv_array, z_score, mean_cross_array


def EAVILS_variance(
    mean_cross_array,
    remove_times=None,
    output_stdv=False,
    maintain_time_dimension=False,
):
    """
    remove_times should be a 1-d boolean array of length Ntimes which will have True for masks and False for no mask.
    This function allows the calculation of the variance using the mean_cross_array.
    The output can be changed to be more like a "standard deviation" array by taking the square root.
    It can also be expanded along its time dimension which can be helpful for shape compatibility
    with other arrays using the maintain_time_dimension=True option
    """
    Ntimes, Ntimes_check, Nfreqs, Npols = mean_cross_array.shape
    if Ntimes != Ntimes_check:
        raise Exception(
            f"Input array has shape {mean_cross_array.shape}, "
            "first two indices should have same length and be the number of included times."
        )
    if remove_times is None:
        remove_times = np.zeros(Ntimes, dtype=bool)
    if len(remove_times) != Ntimes:
        raise Exception(
            f"remove_times has length {len(remove_times)} which differs from value of Ntimes, {Ntimes}"
        )

    used_Ntimes = Ntimes - sum(remove_times)

    dim1_ind, dim2_ind = np.diag_indices(Ntimes, ndim=2)
    keep_times = ~remove_times

    output = (
        np.sum(
            mean_cross_array[dim1_ind[keep_times], dim2_ind[keep_times], :, :], axis=0
        )
        / used_Ntimes
        - np.sum(mean_cross_array[np.ix_(keep_times, keep_times)], axis=(0, 1))
        / used_Ntimes**2
    ) * (used_Ntimes / (used_Ntimes - 1))
    if output_stdv:
        output = np.sqrt(output)
    if maintain_time_dimension:
        output = output * np.full((Ntimes, Nfreqs, Npols), 1)
    return output


# Reads in an options.yml file from the uvfits folder
def options_load(
    uvfits_folder, obs_id, quiet_mode=False, post=False, default_cut_times=(0, -1)
):
    options_name = "options.yml"
    if post:
        options_name = "options_post.yml"
    try:
        with open(
            os.path.join(uvfits_folder, options_name), "r"
        ) as options_reference_file:
            options_reference = yaml.safe_load(options_reference_file)

        date, time = my_utils.obs_id_convert(obs_id)

        if int(obs_id) in options_reference.keys():
            options = options_reference[int(obs_id)]
        elif "default_" + date in options_reference.keys():
            options = options_reference["default_" + date]
        elif "default" in options_reference.keys():
            options = options_reference["default"]
        else:
            options = {}

    except FileNotFoundError as error:

        options = {}
        if default_cut_times is not None:
            options["time_cuts"] = default_cut_times
        if not quiet_mode:
            print(
                f"Exception: no options.yml file found: {error} \nignoring and passing the following: {options}"
            )
    return options


# Runs the things :)
# Main function which checks for existing outputs before running the script
def vis_plotting(
    obs_id,
    plot_types,
    uvfits_folder="",
    output_path="",
    uvd=None,
    load_options=True,
    experimental_output_check=True,
    skip_autos=False,
):

    print(type(uvd))
    plot_types = copy.deepcopy(plot_types)
    if experimental_output_check:
        check_path = os.path.join(output_path, "h5_files")
        check_path = os.path.join(check_path, f"{obs_id}_spectra_data_cross.h5")
        if os.path.isfile(check_path):
            plot_types.remove("spectra")
            print(
                f"[EXPERIMENTAL OUTPUT CHECKING]; found {check_path}, bypassing spectra outputs"
            )

    # In general this
    if plot_types != []:

        obs_id = str(obs_id)

        if load_options:

            options = options_load(uvfits_folder, obs_id)
        else:
            print("load_options set to False, options.yml being ignored")
            options = {}
        '''
        # This will cause the positions of the visibilities in the uv plane to be more organized,
        # takes a long time so avoids unless making movie plots
        if "movie" in plot_types:
            conjugate_bool = True
        else:
            conjugate_bool = False
        '''
        
        # Reads in the object. Note that this uses the use_ss_as_uvd options,
        # which reads in as an undiffed ss object,
        # which functions in the same way as a uvdata object but allows for SSINS to be run without reloading.
        # The uvfits_reader will attempt to find a metafits file in the same folder
        # which it will extract antenna flags from, so try to ensure that uvfits and metafits are in the same folder.
        if uvd is None:
            uvd_cross, uvd_autos = my_utils.uvfits_reader(
                obs_id,
                uvfits_folder,
                conjugate_baselines=conjugate_bool,
                split_autos=True,
                use_ss_as_uvd=True,
            )

        if not os.path.exists(output_path):
            os.makedirs(output_path, mode=0o777)

        # This chunk runs the spectra_maker function which generates SSINS and EAVILS outputs
        ################################################
        if "spectra" in plot_types:
            if not skip_autos:
                spectra_maker(
                    obs_id=obs_id,
                    uvfits_folder=uvfits_folder,
                    output_path=output_path,
                    uvd_cross=uvd_cross,
                    uvd_autos=uvd_autos,
                    options=options,
                )
            else:
                spectra_maker(
                    obs_id=obs_id,
                    uvfits_folder=uvfits_folder,
                    output_path=output_path,
                    uvd_cross=uvd_cross,
                    uvd_autos=None,
                    options=options,
                )
        ################################################

        '''if "movie" in plot_types:
            movie_output_path = os.path.join(output_path, "vis_movies")
            uv_movie_maker(
                obs_id=obs_id,
                uvfits_folder=uvfits_folder,
                output_path=movie_output_path,
                uvd=uvd_cross,
            )'''

        del uvd_cross
        del uvd_autos
        return 0
    else:
        return "already made"
