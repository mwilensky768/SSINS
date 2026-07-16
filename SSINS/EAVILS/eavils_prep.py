import numpy as np


import os
from copy import deepcopy
import gc

import eavils_utils
import eavils_waterfalls

from SSINS import INS
from SSINS import MF

from pyuvdata.parameter import UVParameter




def spectra_maker(
    obs_id,
    input_data_folder,
    output_path,
    shape_dict,
    freq_channel_width,
    pols,
    ss_cross=None,
    ss_autos=None,
    extension='uvfits',
    initial_flags=None
):
    

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



        # This is where the EAVILS information is extracted.
        # Includes intermediate steps as well as final z_score so we can
        # plot intermediate steps and reconstruct the z_scores if necessary

        
###########################################################################################################
        h5_path = os.path.join(output_path, "h5_files")
        if not os.path.exists(h5_path):
            os.makedirs(h5_path, mode=0o777)
        if ss.channel_width[0]!=freq_channel_width:
            raise Exception(f'Frequency width is {ss.channel_width[0]}, not {freq_channel_width} as expected.')

        prefix = os.path.join(h5_path, f"{str(obs_id)}_{freq_channel_width_str(freq_channel_width)}_{bl_type_tag}")
        
        eavils = eavils_waterfalls.EAVILS(ss,spectrum_type=bl_type_tag)
        eavils.write(prefix,clobber=True)

        #Creates and saves an EAVILS divisor storage array
        divisor_storage_array = eavils_waterfalls.build_divisor_storage_array(ss,prefix)
        
        #We don't want to pre-flag our EAVILS plots with SSINS data, so we give ssins_flags=None
        eavils.build_div_and_spectrum(divisor_storage_array,ssins_flags=None)

        
        #Applying diff (sky subtraction) for use in ssins
        ss.diff()

        # eli note: add order parameter here
        ins = INS(ss,spectrum_type=bl_type_tag)
        
        
        ins.write(prefix,clobber=True)

        '''
        #Ensuring frequencies and times which are fully flagged will be applied to ins
        reshaped_initial_flags = np.broadcast_to(initial_flags[ None,:, None], (ins.Ntimes,initial_flags.shape[0], ins.Npols))
        
        initial_ss_flags = reshaped_initial_flags[:,np.newaxis,:,:]
        
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

        
        if ssins_sig_thresh is None:
            sig_thresh = {shape: 5 for shape in shape_dict}
            sig_thresh["narrow"] = 5
            sig_thresh["streak"] = 10
        else:
            sig_thresh=ssins_sig_thresh
            
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
    
        
        ins_copy.write(prefix, output_type='mask', clobber=True)'''

        
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


def freq_channel_width_str(freq_channel_width):
    return f'{int(freq_channel_width/1000)}khz'


def prep(
    obs_id,
    shape_dict,
    input_data_folder,
    output_path,
    freq_channel_width,
    initial_flags,
    pol_dict,
    ss=None,
    output_check=True,
    skip_autos=True,
    extension='uvfits',
    additional_bad_ant_names=[]
):

    pols = list(pol_dict.keys())
    
    if skip_autos:
        bl_type_tag_list = ['cross']
    else:
        bl_type_tag_list = ['cross','auto']
        
    if output_check:
        missing_bool = False
        #for data_type in ['EAVILS_data.h5','divisor_storage.npy','SSINS_data.h5','SSINS_mask.h5']:
        for data_type in ['EAVILS_data.h5','divisor_storage.npy','SSINS_data.h5']:
            for bl_type_tag in bl_type_tag_list:
                check_path = os.path.join(output_path, "h5_files")
                
                
                check_path = os.path.join(check_path, f'{obs_id}_{freq_channel_width_str(freq_channel_width)}_{bl_type_tag}_{data_type}')
                if os.path.isfile(check_path):
                    
                    print(
                        f"OUTPUT CHECKING; found {check_path}, bypassing spectra outputs"
                    )
                else:
                    missing_bool = True
                   
        if not missing_bool:
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

    # This chunk runs the spectra_maker function which generates SSINS and EAVILS outputs
    ################################################
    

    if skip_autos:
        ss_autos_input=None
    else:
        ss_autos_input=ss_autos
        
    spectra_maker(
        obs_id=obs_id,
        input_data_folder=input_data_folder,
        output_path=output_path,
        shape_dict=shape_dict,
        freq_channel_width=freq_channel_width,
        ss_cross=ss_cross,
        ss_autos=ss_autos_input,
        extension=extension,
        pols=pols,
        initial_flags=initial_flags         
    )
    ################################################

    #Being careful with memory hygiene
    del ss_cross
    del ss_autos
    return 0