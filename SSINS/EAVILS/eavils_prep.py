import numpy as np


import os
from copy import deepcopy
import gc

import eavils_utils
import eavils_waterfalls

from SSINS import INS


import time
from datetime import timedelta



def spectra_maker(
    obs_id,
    input_data_folder,
    output_path,
    freq_channel_width,
    pols,
    ss_cross=None,
    ss_autos=None,
    ssins_order=0,
    clobber=False
):
    

    if not os.path.exists(output_path):
        os.makedirs(output_path, mode=0o777)

    spectra_path = os.path.join(output_path, "output_spectra")
    if not os.path.exists(spectra_path):
        os.makedirs(spectra_path, mode=0o777)


    if ss_cross is None and ss_autos is None:
        raise Exception('No input data')
    




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
        

        prefix = os.path.join(h5_path, f"{str(obs_id)}_{freq_channel_width_str(freq_channel_width)}_{bl_type_tag}")
        print("Processing data for EAVILS")
        t0 = time.time()
        eavils = eavils_waterfalls.EAVILS(ss,spectrum_type=bl_type_tag)
        eavils.write(prefix,clobber=clobber)
        t1 = time.time()
        time_diff=t1-t0
        print(f'Time to baseline average, mean subtract and save: {timedelta(seconds=time_diff)}')
        

        t2 = time.time()
        #Creates and saves an EAVILS divisor storage array
        divisor_storage_array = eavils_waterfalls.build_divisor_storage_array(ss,prefix)
        t3 = time.time()
        time_diff=t3-t2
        print(f'Time to create EAVILS divisor and save: {timedelta(seconds=time_diff)}')
        


        
        print('Applying diff (sky subtraction) for use in SSINS')
        t4 = time.time()
        ss.diff()
        t5 = time.time()
        time_diff=t5-t4
        print(f'Time to diff ss object: {timedelta(seconds=time_diff)}')

        print("Processing data for SSINS")
        t6 = time.time()
        ins = INS(ss,spectrum_type=bl_type_tag,order=ssins_order)
       
        ins.write(prefix,clobber=clobber)
        t7 = time.time()
        time_diff=t7-t6
        print(f'Time to create and save ins object: {timedelta(seconds=time_diff)}')

        t8 = time.time()
        #We don't want to pre-flag our EAVILS plots with SSINS data, so we give ssins_flags=None
        eavils.build_div_and_spectrum(divisor_storage_array,ssins_flags=None)
        eavils_waterfalls.plot_maker(eavils=eavils,pols=pols,ins=ins,output_path=spectra_path,name_prefix=str(obs_id),bl_type_tag=bl_type_tag)
        t9 = time.time()
        time_diff=t9-t8
        print(f'Time to create EAVILS waterfall and save plots: {timedelta(seconds=time_diff)}')


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
    input_data_folder,
    output_path,
    freq_channel_width,
    pol_dict=None,
    pol_convention_dict=None,
    ss=None,
    output_check=True,
    skip_autos=True,
    extension='uvfits',
    additional_bad_ant_names=[],
    ssins_order=0,
    return_pol_dict=False,
    clobber=False,
    file_list=None,
    use_file_list=None,
    flag_init=None,
    **read_kwargs
):

    if skip_autos:
        bl_type_tag_list = ['cross']
    else:
        bl_type_tag_list = ['cross', 'auto']

    if output_check:
        missing_bool = False
        for data_type in ['EAVILS_data.h5', 'divisor_storage.npy', 'SSINS_data.h5']:
            for bl_type_tag in bl_type_tag_list:
                check_path = os.path.join(output_path, "h5_files")

                check_path = os.path.join(check_path, f'{obs_id}_{freq_channel_width_str(freq_channel_width)}_{bl_type_tag}_{data_type}')
                if os.path.isfile(check_path):

                    print(
                        f"OUTPUT CHECKING; found {check_path}"
                    )
                else:
                    missing_bool = True

        if not missing_bool:
            if not clobber:
                print(f'All output files found, skipping. Set clobber=True to overwrite instead.')
                return 0
            else:
                print(f'All output files found. clobber is set to True, overwriting existing files.')

    obs_id = str(obs_id)

    # Reads in the object. Note that data reads in as an undiffed ss object,
    # which functions in the same way as a uvdata object but allows for SSINS to be run without reloading.
    # eavils_utils.reader will attempt to find a metafits file in the same folder
    # which it will extract antenna flags from, so try to ensure that uvfits and metafits are in the same folder.
    if ss is None:
        ss_cross, ss_autos = eavils_utils.reader(
            obs_id,
            input_folder=input_data_folder,
            split_autos=True,
            additional_bad_ant_names=additional_bad_ant_names,
            extension=extension,
            keep_autos=skip_autos,  # When keep_autos is set to False, ss_autos will just be None
            file_list=file_list,
            use_file_list=use_file_list,
            flag_init=flag_init,
            **read_kwargs
        )

    if not os.path.exists(output_path):
        os.makedirs(output_path, mode=0o777)

    if pol_convention_dict is None:
        pol_dict_from_file = {pols: index for index, pols in enumerate(ss_cross.get_pols())}
    else:
        pol_dict_from_file = {''.join([pol_convention_dict[pol] for pol in pols]): index for index, pols in enumerate(ss_cross.get_pols())}

    if pol_dict is None:
        pol_dict = pol_dict_from_file
        print(f'pol_dict inferred from file: {pol_dict}')
    elif pol_dict != pol_dict_from_file:
        raise Exception(f'input pol_dict is {pol_dict}, while the file\'s ordering is {pol_dict_from_file}, derived from ss_cross.get_pols()={ss_cross.get_pols()}, using pol_convention_dict={pol_convention_dict}. Please ensure the input pol_dict is consistent with this. Use the pol_convention_dict argument if different naming conventions are used.')

    pols = list(pol_dict.keys())

    if ss_cross.channel_width[0] != freq_channel_width:
        raise Exception(f'Frequency width is {ss.channel_width[0]}, not {freq_channel_width} as expected.')

    if skip_autos:
        ss_autos_input = None
    else:
        ss_autos_input = ss_autos

    spectra_maker(
        obs_id=obs_id,
        input_data_folder=input_data_folder,
        output_path=output_path,
        freq_channel_width=freq_channel_width,
        ss_cross=ss_cross,
        ss_autos=ss_autos_input,
        extension=extension,
        pols=pols,
        clobber=clobber
    )

    del ss_cross
    del ss_autos
    if return_pol_dict:
        return pol_dict