import numpy as np


import os
from copy import deepcopy
import gc

import eavils_utils
import eavils_waterfalls

from SSINS import INS
from SSINS import MF

from pyuvdata.parameter import UVParameter




def mwa_spectra_maker(
    obs_id,
    input_data_folder,
    output_path,
    shape_dict_name='MWA_high',
    ss_cross=None,
    ss_autos=None,
    extension='uvfits',
    freq_channel_width=80000
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


    if ss_cross is None and ss_autos is None:
        raise Exception('No input data')
    print("Preparing data")




    # This section just sets up proper tags and settings for auto and cross spectra
    bl_type_tag_list = []
    if ss_autos is not None:
        bl_type_tag_list.append("auto")

    if ss_cross is not None:
        bl_type_tag_list.append("cross")

    for bl_type_tag in bl_type_tag_list:
        if bl_type_tag == "auto":
            auto_bool = True
            ss = ss_autos
            del ss_autos

        elif bl_type_tag == "cross":
            auto_bool = False
            ss = ss_cross
            del ss_cross

        


        pols = ["XX", "YY", "XY", "YX"]

        # This is where the EAVILS information is extracted.
        # Includes intermediate steps as well as final z_score so we can
        # plot intermediate steps and reconstruct the z_scores if necessary

        
###########################################################################################################
        h5_path = os.path.join(output_path, "h5_files")
        if not os.path.exists(h5_path):
            os.makedirs(h5_path, mode=0o777)
        if ss.channel_width[0]!=freq_channel_width:
            raise Exception(f'Frequency width is {ss.channel_width[0]}, not {freq_channel_width} as expected.')
        channel_width = f"{int(ss.channel_width[0]/1000)}khz"
        prefix = os.path.join(h5_path, f"{str(obs_id)}_{channel_width}_{bl_type_tag}")
        
        eavils = eavils_waterfalls.EAVILS(ss,spectrum_type=bl_type_tag)
        eavils.write(prefix,clobber=True)

        divisor_storage_array = eavils_waterfalls.build_divisor_storage_array(ss,prefix)
        
        #We don't want to pre-flag our EAVILS plots with SSINS data, so we give ssins_flags=None
        eavils.build_div_and_spectrum(divisor_storage_array,ssins_flags=None)

        
        #Applying diff (sky subtraction) for use in ssins
        ss.diff()


        ins = INS(ss,spectrum_type=bl_type_tag)

        ins.write(prefix,clobber=True)
        
        #Ensuring frequencies and times which are fully flagged will be applied to ins
        initial_flags = eavils_utils.coarse_band_flagging(
            Nfreqs=len(ins.freq_array),
            coarse_band_count=24,
            flag_centers=mwax_bool,
            flag_edges=True,
            Ntimes=ins.Ntimes,
            Npols=ins.Npols
       )
        
        initial_ss_flags = initial_flags[:,np.newaxis,:,:]
        
        initial_ss_flags = np.ones(
            (
                ss.Ntimes,
                ss.Nbls,
                ss.Nfreqs,
                ss.Npols)
            ,dtype=bool
        )*initial_ss_flags
        
        initial_ss_flags = np.reshape(initial_ss_flags,(ss.Ntimes*ss.Nbls,ss.Nfreqs,ss.Npols))

        
        ss.apply_flags(flag_choice="custom", custom=initial_ss_flags)


        ins_copy = INS(ss,spectrum_type=bl_type_tag)
        shape_dict = eavils_utils.get_shape_dict(shape_dict_name)
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
    
        
        ins_copy.write(prefix, output_type='mask', clobber=True)

        
        eavils_waterfalls.plot_maker(eavils=eavils,ins=ins,pols=pols,output_path=spectra_path,name_prefix=str(obs_id),bl_type_tag=bl_type_tag)


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




# Main function which checks for existing outputs before running the script
def vis_plotting(
    obs_id,
    input_data_folder="",
    output_path="",
    ss=None,
    output_check=True,
    skip_autos=True,
    extension='uvfits',
    freq_channel_width=80000,
    additional_bad_ant_names=[]
):

    
    
    if output_check:
        check_path = os.path.join(output_path, "h5_files")
        
        
        check_path = os.path.join(check_path, f'{obs_id}_{int(freq_channel_width/1000)}khz_cross_EAVILS_data.h5')
        if os.path.isfile(check_path):
            
            print(
                f"OUTPUT CHECKING; found {check_path}, bypassing spectra outputs"
            )
            return 0




    obs_id = str(obs_id)

    
    
    # This will cause the positions of the visibilities to all be in the same half of the uv-plane along some dividing line. SLow and unnecessary for EAVILS so disabled.


    
    
    
    # Reads in the object. Note that data reads in as an undiffed ss object,
    # which functions in the same way as a uvdata object but allows for SSINS to be run without reloading.
    # eavils_utils.reader will attempt to find a metafits file in the same folder
    # which it will extract antenna flags from, so try to ensure that uvfits and metafits are in the same folder.
    if ss is None:
        ss_cross, ss_autos = eavils_utils.reader(
            obs_id,
            input_folder=input_data_folder,
            split_autos=True,
            additional_bad_ant_names=additional_bad_ant_names
        )

    if not os.path.exists(output_path):
        os.makedirs(output_path, mode=0o777)

    # This chunk runs the mwa_spectra_maker function which generates SSINS and EAVILS outputs
    ################################################
    

    if not skip_autos:
        mwa_spectra_maker(
            obs_id=obs_id,
            input_data_folder=input_data_folder,
            output_path=output_path,
            ss_cross=ss_cross,
            ss_autos=ss_autos,
            extension=extension,
            freq_channel_width=freq_channel_width
        )
    else:
        mwa_spectra_maker(
            obs_id=obs_id,
            input_data_folder=input_data_folder,
            output_path=output_path,
            ss_cross=ss_cross,
            ss_autos=None,
            extension=extension,
            freq_channel_width=freq_channel_width
        )
    ################################################

    #Being careful with memory hygiene
    del ss_cross
    del ss_autos
    return 0