import sys


import eavils_waterfalls as wtrf
from SSINS import INS
from SSINS import MF
from pyuvdata import UVFlag

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import eavils_utils
import scipy
import os
import yaml
from copy import deepcopy
import pandas as pd
import bidict
import itertools
import astropy.io.fits as fits





def freq_range_sort(shape_dict):
    '''This function simply sorts the frequency ranges defined in a shape_dict to be ordered in order of their minimum frequency. Should be used in all instances where iterating through frequency ranges is necessary.'''
    sorted_freq_ranges = []
    sorted_freq_mins = [shape_dict[freq_range][0] for freq_range in shape_dict.keys()]
    sorted_freq_mins = sorted(sorted_freq_mins)
    for freq_min in sorted_freq_mins:
        for freq_range in shape_dict.keys():
            if shape_dict[freq_range][0] == freq_min:
                sorted_freq_ranges.append(freq_range)
    return sorted_freq_ranges,sorted_freq_mins



def get_filenames(input_directory,suffix_dict,list_or_file=None,allowed_missing_fraction=0,return_missing_list=False):
    '''Takes an input directory and identifies all files which have end with the given suffixes that are in a check list. 
    
    Expects files to be in a <obs_tag>_<suffix> format. 
    Compares obs_tag's of these files to a .txt list file with \n as the seperator
    Returns a dict that can translate from obs_tag to different file names.'''
    
    full_file_list = os.listdir(input_directory)
    if type(list_or_file) is str:
        with open(list_or_file,'r') as file:
            observation_check_list = file.read().split('\n')
        observation_check_list = [obs_tag for obs_tag in observation_check_list if obs_tag.rstrip()!='']
    elif type(list_or_file) is list:
        observation_check_list = list_or_file
    elif list_or_file is None:
        arbitrary_suffix = next(suffix_dict[x] for x in suffix_dict)
        observation_check_list = []
        for filename in full_file_list:
            if filename.endswith(arbitrary_suffix):
                obs_tag = filename.split('_')[0]
                observation_check_list.append(obs_tag)
    else:
        raise Exception(f'incorrect type for list_or_file: {type(list_or_file)}, expected str or list.')
    observation_check_list = sorted(observation_check_list)
    
    filename_dict = {}
    missing_list = []
    for obs_tag in observation_check_list:
        filename_dict[obs_tag] = {}
        for data_tag, suffix in suffix_dict.items():
            filename = f'{obs_tag}_{suffix}'
            if filename in full_file_list:
                filename_dict[obs_tag][data_tag] = os.path.join(input_directory,filename)
            else:
                missing_list.append(obs_tag)
                break

    for obs_tag in missing_list:
        del filename_dict[obs_tag]
    if len(missing_list)>0:
        if len(missing_list)/len(observation_check_list) > allowed_missing_fraction:
            raise Exception(f'missing files in {input_directory}; expected file num.: {len(observation_check_list)}, found {len(filename_dict)}. Missing: {missing_list}')
        elif allowed_missing_fraction<1: 
            print(f'WARNING: missing files in {input_directory}; expected files num.: {len(observation_check_list)}, found {len(filename_dict)}. Missing: {missing_list}')
    if return_missing_list:
        return filename_dict,missing_list
    else:
        return filename_dict
    


def title_gen(time_dim,freq_dim):
    '''Generates the 'title' which references the number of times and frequencies in an averaging block'''
    return f'Tdim{time_dim}_Fdim{freq_dim}'




def process_data(
    list_file,
    input_directory,
    output_directory,
    time_dim,
    freq_dim,
    pol_bidict,
    shape_dict,
    initial_freq_flags,
    metafits_folder,
    suffix_add,
    pre_flag_on_SSINS_masks=True,
    clobber=False,
    return_output=False,
    allowed_missing_fraction=0,
    ssins_sig_thresh=None,
    remove_edges_of_ranges=True
):
    '''The primary function for processing raw data (individual observation EAVILS h5 files) into a larger set. Averages the data in blocks and calculates the frequency across channels. Should be run seperately for each combinations of night and sky field, each of which should be associated with a seperate list_file
    
    list_file is an \n seperated text file with a list of obs_tag's associated with a particular night and sky field observation
    input_directory defines where to find the EAVILS uvflag files and divisor storage files. They will be compared to obs_tag's in the list_file
    output_directory defines where to output the processed data.
    time_dim defines the number of time steps averaged across. However, if the time dimension of a given spectrum doesn't divide evenly by the time_dim the remainder is averaged over. In other words, if there are 52 time indices, and the time_dim is 8, there will be 6 blocks of dimension 8 to be averaged over, and one remainder block of dimension 4 which will be averaged over.
    freq_dim defines the frequencies to be averaged across in blocks. As it stands, the averaging skips over any initially flagged channels (such as the coarse-band lines of the MWA). Works similarly to the time_dim but is contained within a frequency channel defined by shape_dict
    pol_bidict just defines the relationship between the names of the polarizations and their indices, (e.g. 'XX' is associated with an index of 0)
    shape_dict defines the possible channels for RFI. In the MWA context this equates to digital TV channels. It's also helpful to include a "clean" channel, where we don't expect digital TV RFI, which can serve as a control for the other channels. However, it's important to note that it's possible to have other types of RFI in this channel, such as narrow-band.
    initial_freq_flags should be a frequency mask, which has True for flagged channels and False for unflagged channels. This should have the same dimensions as eavils.freq_array or ins.freq_array
    pre_flag_on_SSINS_masks uses the previously generated SSINS mask for each EAVILS spectrum to exclude that data from calculations of the mean and estimated standard deviation. It assumes the SSINS masks have been propagated in polarization which is generally the case.
    clobber overwrites existing processed h5 files where they exist if True.'''

    
    
    #This chunk sets up some initial variables derived directly from input params.

    suffix_dict = {'EAVILS':'EAVILS_data.h5','SSINS_data':'SSINS_data.h5','divisor_storage_array':'divisor_storage.npy'}
    suffix_dict = {key:f'{suffix_add}_{suffix_dict[key]}' for key in suffix_dict.keys() }
    
    list_title = list_file.split('/')[-1].split('.')[0]
    output_sub_directory = os.path.join(output_directory,list_title)
    title = title_gen(time_dim,freq_dim)

    if not os.path.exists(output_sub_directory):
        os.makedirs(output_sub_directory, mode=0o777)
 
 


    existing_output_files_dict = get_filenames(
        output_sub_directory,
        suffix_dict={'var':f'{title}_EAVILS_variance.h5','flag':f'{title}_merged_SSINS_flags.h5'},
        list_or_file=list_file,
        allowed_missing_fraction=1
    )
    existing_file_obs_tags = list(existing_output_files_dict.keys())
    


    
    sorted_freq_ranges,sorted_freq_mins = freq_range_sort(shape_dict)
    
    filename_dict = get_filenames(
        input_directory,
        suffix_dict=suffix_dict,
        list_or_file=list_file,
        allowed_missing_fraction=allowed_missing_fraction
    )
    first_iteration_bool=True



    # This loop iterates through the obs_tag's and filenames from the list_file present in the input_directory
    processed_count = 0
    for obs_tag in filename_dict.keys():
        if obs_tag in existing_file_obs_tags and clobber==False:
            continue
        
        # This block reads in data from files and sets up data objects
        eavils = wtrf.EAVILS(filename_dict[obs_tag]['EAVILS'])
        if first_iteration_bool:
            instrument_name = eavils.telescope.instrument
        else:
            if eavils.telescope.instrument!=instrument_name:
                raise Exception(f'Data from different instruments, {instrument_name} and {eavils.telescope.instrument}')
            
        divisor_storage_array = np.load( filename_dict[obs_tag]['divisor_storage_array'])


        ins = run_ssins(
            ssins_filename=filename_dict[obs_tag]['SSINS_data'],
            initial_freq_flags=initial_freq_flags,
            shape_dict=shape_dict,
            ssins_sig_thresh=ssins_sig_thresh
        )
        
        if pre_flag_on_SSINS_masks:
            eavils.build_div_and_spectrum(divisor_storage_array,ssins_flags = ins.mask_to_flags()) 
        else:
            eavils.build_div_and_spectrum(divisor_storage_array)


        
        # Builds the range_ind_dict which identifies freq indices in each frequency range but not in initial_freq_flags
        if first_iteration_bool:
            first_obs_tag = obs_tag
            freq_array_default = eavils.freq_array
            range_ind_dict = {}
            if len(freq_array_default)!=len(initial_freq_flags):
                raise Exception(f'Length of freq_array {len(freq_array_default)} is not equal to length of the initial_freq_flags array {len(initial+freq_flags)}')
            for freq_range in sorted_freq_ranges:
                    
                freq_range_inds = eavils_utils.freq_ind_finder(freqs=eavils.freq_array,ranges=[shape_dict[freq_range]])
                
                freq_range_inds = [ind for ind in freq_range_inds if initial_freq_flags[ind]==False]

                if remove_edges_of_ranges:
                    freq_range_inds = freq_range_inds[1:-1] 
                range_ind_dict[freq_range] = freq_range_inds

            
            
        # Ensures that the frequency array for this data are consistent with that found in the first file.
        if np.any(eavils.freq_array!=freq_array_default):
            raise Exception(f'freq_array from {filename} does not match the freq_array from the first file processed, for observation {obs_tag}. Please ensure that all obervations in {list_file} are from the same set of observations.')

        


            
        Ntime_blocks = int(np.ceil(eavils.Ntimes/time_dim))
        Nfreq_ranges = len(sorted_freq_ranges)
        
        
        var_metric_array = np.full((Ntime_blocks,Nfreq_ranges,eavils.Npols),np.nan)
        merged_flags_array = np.full((Ntime_blocks,Nfreq_ranges,eavils.Npols),np.nan)
        
        #This loop over frequency ranges (e.g. DTV chan's) does block averaging and variance calculations.
        for f_range_ind, freq_range in enumerate(sorted_freq_ranges):
            
            superpixel_eavils=[]
            superpixel_flags=[]

            # Makes sub-arrays w/ freq. indices from range_ind_dict
            freq_range_inds = range_ind_dict[freq_range]
            eavils_sub_array = eavils.spectrum[:,freq_range_inds,:]
            ssins_flag_sub_array = ins.mask_to_flags()[:,freq_range_inds,:]

            # Block averaging is done in this loop over pols
            for pol_ind,pol in pol_bidict.items():                                           

                superpixel_eavils.append(
                    block_average(
                        eavils_sub_array[:,:,pol_ind],
                        time_dim,freq_dim,
                        return_same_shape=False,
                        scale_by_sqrt_n=True
                    )
                )
                superpixel_flags.append(
                    block_average(
                        ssins_flag_sub_array[:,:,pol_ind],
                        time_dim,
                        freq_dim,
                        return_same_shape=False,
                        scale_by_sqrt_n=False,
                        treat_as_boolean=True
                    )
                )
    
    
            #Making into numpy arrays for processing
            superpixel_eavils = np.array(superpixel_eavils)
            superpixel_flags = np.array(superpixel_flags)

            #Fixing the order of axes to maintain the time, frequency, polarization convention used throughout
            superpixel_eavils = np.moveaxis(superpixel_eavils,0,-1)
            superpixel_flags = np.moveaxis(superpixel_flags,0,-1)


            #The per-time chunk, per frequency-channel, per polarization variance across frequency
            var_metric_array[:,f_range_ind,:]=np.var(superpixel_eavils,axis=1,ddof=1)
            # blocked SSINS flags - if at least one flag, flag whole block
            merged_flags_array[:,f_range_ind,:]=np.any(superpixel_flags,axis=1)

        merged_flags_array = merged_flags_array.astype(bool)

            
        # This chunk saves out the blocked variance and SSINS flag to a UVFlag file
        uvf_metric = UVFlag()
        uvf_flag = UVFlag()
        # Building frequency data for UVFlag
        biggest_freq = np.max(eavils.freq_array)
        smallest_freq = np.min(eavils.freq_array)
        freq_range_lims_array = [shape_dict[freq_range] for freq_range in sorted_freq_ranges]
        freq_range_lims_array = [[max(freq_range_lims[0],smallest_freq),min(freq_range_lims[1],biggest_freq)] for freq_range_lims in freq_range_lims_array]
        freq_range_channel_width = [freq_range_lims[1]-freq_range_lims[0] for freq_range_lims in freq_range_lims_array]
        freq_range_centers_array = [np.mean(freq_range_lims) for freq_range_lims in freq_range_lims_array]
        
        
        if eavils.Nspws>1:
            raise Exception(f'Nspws of {eavils.Nspws}. Multiple spectral windows not currently implemented for eavils_processing')
        else:
            range_flex_spw_id_array = [0 for f in range(len(sorted_freq_ranges))]
        
        # Time data for UVFlag
        averaged_lst_array = np.array([np.mean(eavils.lst_array[i+time_dim]) for i in range(Ntime_blocks)])
        averaged_time_array = np.array([np.mean(eavils.time_array[i+time_dim]) for i in range(Ntime_blocks)])
        if np.any(eavils.weights_array != eavils.weights_array[0,0,0]):
            raise Exception('weights_array has at least one non-identical value. This is not currently handled in eavils_processing')
        else:
            reshaped_weights_array = np.full((Ntime_blocks,len(sorted_freq_ranges),eavils.Npols),eavils.weights_array[0,0,0])
        #Inputting all required uvf parameters
        for uvf_type in ['metric','flag']:
            if uvf_type=='metric':
                uvf = uvf_metric
            elif uvf_type =='flag':
                uvf = uvf_flag
            uvf.Nblts=None    
            uvf.Nfreqs=Nfreq_ranges          
            uvf.Npols=eavils.Npols
            uvf.Nspws=eavils.Nspws            
            uvf.Ntimes=Ntime_blocks            
            uvf.channel_width=freq_range_channel_width
            uvf.flex_spw_id_array=range_flex_spw_id_array
            uvf.freq_array=freq_range_centers_array
            if uvf_type=='metric':
                uvf.history=str(eavils.history+
                    f'Time averaged over {time_dim} integrations.'+
                    f' Variance calculated across the following channels {sorted_freq_ranges}.'
                )
            elif uvf_type =='flag':
                uvf.history=str(eavils.history+
                    f'Time averaged over {time_dim} integrations.'+
                    f' Merged SSINS flags taken from the following channels {sorted_freq_ranges}.'
                )
            uvf.label=eavils.label
            uvf.lst_array=averaged_lst_array
            uvf.mode=uvf_type
            uvf.polarization_array=eavils.polarization_array
            uvf.spw_array=eavils.spw_array
            uvf.telescope=eavils.telescope
            uvf.time_array=averaged_time_array
            uvf.type='waterfall'
            uvf.weights_array=reshaped_weights_array
            if uvf_type=='metric':
                uvf.metric_array = var_metric_array    
                save_name = f'{obs_tag}_{title}_EAVILS_variance.h5'

            elif uvf_type=='flag':
                uvf.flag_array = merged_flags_array
                save_name = f'{obs_tag}_{title}_merged_SSINS_flags.h5'
                
            save_name = os.path.join(output_sub_directory,save_name)   
            try:
                uvf.write(save_name,clobber=clobber)
            except ValueError as e:
                print(f'{e}; change clobber to True to overwrite')
        processed_count+=1
        first_iteration_bool=False
        
    pointing_yaml_name = f'{title}_ptng_info.yml'
    pointing_yaml_name = os.path.join(output_sub_directory,pointing_yaml_name)
    if not os.path.exists(pointing_yaml_name) or clobber==True:
        if instrument_name=='MWA':
            #Generate a pointing info dict to save per obs_id pointing info for MWA data
            pointing_info_dict = mwa_pointing_identification([metafits_folder], obs_id_list = list(filename_dict.keys()))
        
            with open(pointing_yaml_name, 'w') as file:
                yaml.safe_dump(pointing_info_dict, file, sort_keys=False)

    
    print(f'{processed_count} observations processed')  

    
    
    
        



    

def mwa_pointing_identification(
    metafits_folders, obs_id_list
, include_pointing_int=True):

    #Returns pointing information for each obs_id in a dictionary format. Right now is specific to the MWA for EOR-0 and EOR-1 fields. Should be made more general

    if not isinstance(obs_id_list[0], str):
        print(
            "Warning: Correcting type of obs_id_list argument of raw_pointing_list to string"
        )
        obs_id_list = [str(obs_id) for obs_id in obs_id_list]
    pointing_info_dict = {}
    for metafits_folder in metafits_folders:
        for file in os.listdir(metafits_folder):
            if ".metafits" in file:
                
                obs_id = file.split(".")[0]

                if obs_id in obs_id_list:
                    metafits = fits.open(os.path.join(metafits_folder, file))
                    ra = metafits["primary"].header["RA"]
                    dec = metafits["primary"].header["DEC"]
                    alt = metafits["primary"].header["ALTITUDE"]
                    az = metafits["primary"].header["AZIMUTH"]
                    if include_pointing_int:
                        pointing = eavils_utils.mwa_pointings(az, alt, tolerance=0.01)
                        pointing_info_dict[obs_id] = {'ra':ra, 'dec':dec, 'alt':alt, 'az':az, 'pointing':pointing}
                    else:
                        pointing_info_dict[obs_id] = {'ra':ra, 'dec':dec, 'alt':alt, 'az':az}

    missing_list = []
    for obs_id in obs_id_list:
        if obs_id not in pointing_info_dict.keys():
            missing_list.append(obs_id)
    if len(missing_list) > 0:
        raise Exception(f"Missing the following obs_id's metafits files: {missing_list}")

    
    
    return pointing_info_dict






def block_average(arr, block_rows, block_cols, return_same_shape=False, scale_by_sqrt_n=False,treat_as_boolean=False):
    #Block averages a 2-dimensional array into blocks of specified size. Default behavior is to average. 
    #return_same_shape allows you to choose to return the array with average values recast to original shape.
    #scale_by_sqrt_n allows you to multiply each average by the number of included data points to scale noise correctly (only applies if treat_as_boolean is False)
    #treat_as_boolean is used for a boolean array, if enabled will just return True if any value in a given block is True
    rows, cols = arr.shape

    # Compute padding
    pad_rows = (-rows) % block_rows
    pad_cols = (-cols) % block_cols

    # Pad array and mask
    arr_padded = np.pad(arr, ((0, pad_rows), (0, pad_cols)), mode='constant', constant_values=0)
    mask = np.pad(np.ones_like(arr), ((0, pad_rows), (0, pad_cols)), mode='constant', constant_values=0)

    padded_rows, padded_cols = arr_padded.shape

    # Reshape into blocks
    arr_blocks = arr_padded.reshape(
        padded_rows // block_rows, block_rows,
        padded_cols // block_cols, block_cols
    )
    mask_blocks = mask.reshape(
        padded_rows // block_rows, block_rows,
        padded_cols // block_cols, block_cols
    )

    if treat_as_boolean:
        if np.any(arr_blocks.sum(axis=(1, 3))<0):
            raise Exception('Unexpected value of array block with treat_as_boolean enabled. Please ensure that array consists only of boolean values or 1s and 0s.')
        
        block_values = arr_blocks.sum(axis=(1, 3))>0
    else:
        block_sums = arr_blocks.sum(axis=(1, 3))
        block_counts = mask_blocks.sum(axis=(1, 3))

        # Compute mean and scale if requested
        block_means = np.divide(block_sums, block_counts, where=block_counts != 0)
    
        if scale_by_sqrt_n:
            block_means *= np.sqrt(block_counts)
        block_values = block_means
    if not return_same_shape:
        return block_values

    # Expand and trim to original shape
    expanded = np.repeat(
        np.repeat(block_values, block_rows, axis=0),
        block_cols, axis=1
    )
    return expanded[:rows, :cols]

def add_1D_mask(array, mask):
    mask= mask[np.newaxis,:,np.newaxis]
    mask = mask*np.ones(array.shape)
    return np.ma.masked_array(data = array, mask = mask)





def create_data_arrays(processed_data_directory,list_titles,time_dim,freq_dim,shape_dict,sky_field_list_association,pol_subtraction_order,add_pointing_dict):
    #Turns data in the processed data h5 files into per-obs_id arrays, calculates the pol_sub values.
    #processed_data_directory is the parent directory for the processed data
    #list_titles is the set of list_titles associated with each combination of night and field of observation
    #time_dim and freq_dim are as described above, size of averaging blocks, here are just important for selecting the correct processed file
    #shape_dict gives the frequency channels where we expect our DTV type RFI
    #sky_field_association associates list_titles with sky_fields. This is important for getting statistics right down the road.
    #pol_subtraction_order gives the order in which polarizations are subtracted. For consistency, the pol_sub arrays are kept in the same shape as the other arrays despite only the first polarization subtraction combination being used. In general, it is only sensible to subtract polarizations with similar underlying statistics from each other, so e.g. EE-NN or EN-NE, which is why we don't include combinations such as (0,2).
    
    #The function returns:
    #an array_dict, which contains the following, all given per time block, per frequency channel, and, in the first two cases, per polarization: variance, the SSINS mask (True if any flagged data in block), and the pol_sub values (the difference of the variance between sets of polarization). 
    #a dof_ref_dict or degrees of freedom reference dict, just giving the number of blocks where variances is taken over per frequency channel
    #a sky_field_dict which associates each obs_id with its sky_field
    #a source_list_dict which associates each obs_id with its source_list
    ###################################
    title=title_gen(time_dim,freq_dim)
    sorted_freq_ranges,sorted_freq_mins = freq_range_sort(shape_dict)
    array_dict = {'variance':{},'reshaped_SSINS_mask':{},'pol_sub':{}}
    dof_ref_dict = {}
    
    sky_field_dict = {}
    source_list_dict = {}
    if add_pointing_dict:
        combined_pointing_info_dict={}
    print('creating data arrays')
    first_time_bool=True
    for list_title in list_titles:

        sky_field = sky_field_list_association[list_title]
        print(list_title)

        processed_file_sub_directory = f'{processed_data_directory}/{list_title}'
        if add_pointing_dict:
            with open(f'{processed_file_sub_directory}/{title}_ptng_info.yml') as yaml_file:
                partial_ptng_dict = yaml.safe_load(yaml_file)
            combined_pointing_info_dict = combined_pointing_info_dict | partial_ptng_dict
        for file_name in os.listdir(processed_file_sub_directory):
            obs_tag = file_name.split('_')[0]
            
        
        filename_dict = get_filenames(
            processed_file_sub_directory,
            suffix_dict={'var':f'{title}_EAVILS_variance.h5','flag':f'{title}_merged_SSINS_flags.h5'},
            allowed_missing_fraction=0)
        
        for obs_tag in filename_dict.keys():
            sky_field_dict[obs_tag] = sky_field
            source_list_dict[obs_tag] = list_title
            
            var_info = UVFlag()
            var_info.read(filename_dict[obs_tag]['var'])
            flag_info = UVFlag()
            flag_info.read(filename_dict[obs_tag]['flag'])
            
            var_plot_array = var_info.metric_array
            reshaped_flags_array = flag_info.flag_array
        
            pol_sub_array = np.full(var_plot_array.shape,np.nan)
            for pseudo_pol_ind,(polA_ind,polB_ind) in enumerate(pol_subtraction_order):
                pol_sub_array[:,:,pseudo_pol_ind] = var_plot_array[:,:,polA_ind] - var_plot_array[:,:,polB_ind]
                
            array_dict['variance'][obs_tag] = var_plot_array
            array_dict['pol_sub'][obs_tag]  = pol_sub_array
            array_dict['reshaped_SSINS_mask'][obs_tag] = reshaped_flags_array
    if add_pointing_dict:
        return array_dict,sky_field_dict,source_list_dict, combined_pointing_info_dict
    else:
        return array_dict,sky_field_dict,source_list_dict


 
            
    
    ##########################################################################


def create_data_frame(pol_bidict,shape_dict,array_dict,sky_field_dict,source_list_dict,pointing_info_dict):
    #pol_bidict associates the polarizations with their indices
    #shape_dict gives the frequency channels where we expect our DTV type RFI
    #array_dict is the result of the create_array_dict function explicated above
    #sky_field_dict and source_list_dict are similarly explicated above
    
    #Returns a pandas dataframe which collects the important data for each measurement into a single pandas dataframe. In other words, for each frequency and time, there is a row with all the input data.
    sorted_freq_ranges,sorted_freq_mins = freq_range_sort(shape_dict)
    obs_tag_list = list(array_dict['variance'].keys())
    datafr_dict={'obs_tag':[],'sky_field':[],'t_block_ind':[],'pointing':[],'pol_sub':[],'SSINS_flagged':[],'pointing_flagged':[],'source_list':[]}
    for pol in pol_bidict.inverse.keys():
        datafr_dict[f'variance_{pol}']=[]
    for freq_range in sorted_freq_ranges:
        datafr_dict['freq_range']=[]
    
    for obs_tag in obs_tag_list:
        Ntime_blocks,_,_ = array_dict['variance'][obs_tag].shape
        for freq_range_ind, freq_range in enumerate(sorted_freq_ranges):
            sky_field = sky_field_dict[obs_tag]
            source_list = source_list_dict[obs_tag]
            datafr_dict['obs_tag']+=[obs_tag for i in range(Ntime_blocks)]
            datafr_dict['sky_field']+=[sky_field for i in range(Ntime_blocks)]
            datafr_dict['source_list']+=[source_list for i in range(Ntime_blocks)]
            datafr_dict['pointing']+=[pointing_info_dict[obs_tag]['pointing'] for i in range(Ntime_blocks)]
            datafr_dict['freq_range']+=[freq_range for i in range(Ntime_blocks)]
            datafr_dict['t_block_ind']+=[i for i in range(Ntime_blocks)]
            
            for pol_ind,pol in pol_bidict.items():
                datafr_dict[f'variance_{pol}']+=list(array_dict['variance'][obs_tag][:,freq_range_ind,pol_ind])
    
            
            datafr_dict[f'SSINS_flagged']+=list(np.bool_(array_dict['reshaped_SSINS_mask'][obs_tag][:,freq_range_ind,0]))
    
            
            pol_sub_mini_list = (array_dict['pol_sub'][obs_tag][:,freq_range_ind,0])
            
            datafr_dict['pol_sub']+=list(pol_sub_mini_list)
            datafr_dict['pointing_flagged']+=[False for i in range(Ntime_blocks)]
        
        
        
    
    datafr=pd.DataFrame(datafr_dict)
    return datafr



def add_pol_sub_stdv(datafr,estimated_pol_sub_stdv,threshold):
    #This adds the column for the estimated pol_sub standard deviation, necessary for scaling thresholds. This must be calculated independently for each sky field and array configuration. Its critical that this is done correctly for good results.
    datafr=deepcopy(datafr)
    datafr['pol_sub_stdv'] = np.nan
    
    for sky_field in list(estimated_pol_sub_stdv.keys()):
        for freq_range in list(estimated_pol_sub_stdv[sky_field].keys()):
    
            selected_inds = datafr.query(f'freq_range == \'{freq_range}\' & sky_field == \'{sky_field}\'').index
            datafr.loc[selected_inds,'pol_sub_stdv'] = estimated_pol_sub_stdv[sky_field][freq_range]
    missing_values = np.sum(pd.isna(datafr['pol_sub_stdv']))
    if missing_values>0:
        raise Exception(f'Missing {missing_values} values. Please ensure that estimated_pol_sub_stdv input contains all sky_fields and freq_ranges included in the datafr')

    datafr['pol_sub_flagged'] = np.abs(datafr['pol_sub'])>(datafr['pol_sub_stdv']*threshold)
    return datafr



def find_per_pointing_stats(datafr,shape_dict,list_titles,threshold):
    # This calculates per pointing stats for each list_title, including how many pol_sub measurements crossed the thrshold set, as well as the scaled absolute mean (scaled according to the expected mean and variance for a folded normal distribution), and then the same calculation with threshold crossing measurements removed.
    sorted_freq_ranges,sorted_freq_mins = freq_range_sort(shape_dict)

    all_pointings = np.unique(datafr['pointing'])

    
    folded_mean = np.sqrt(2/np.pi)
    folded_var = 1-2/np.pi
    
    
    per_pointing_stats = {}
    for list_title in list_titles:
        per_pointing_stats[list_title] = {}
        for freq_range in sorted_freq_ranges:
            per_pointing_stats[list_title][freq_range] = {}
            for pointing in all_pointings:
                per_pointing_stats[list_title][freq_range][pointing] = {}
                
                query_string = f'{'freq_range'} == \'{freq_range}\'&{'pointing'} == {pointing}& SSINS_flagged == False & source_list == \'{list_title}\''
                pointing_pol_sub_series = np.array(datafr.query(query_string)['pol_sub']/datafr.query(query_string)['pol_sub_stdv'])
                meas_count = len(pointing_pol_sub_series)
                abs_mean = np.sqrt(meas_count/folded_var)*(np.mean(np.abs(pointing_pol_sub_series))-folded_mean)
                per_pointing_stats[list_title][freq_range][pointing]['abs_mean'] = abs_mean
                
                query_string = f'{'freq_range'} == \'{freq_range}\'&{'pointing'} == {pointing}& SSINS_flagged == False & pol_sub_flagged == False & source_list == \'{list_title}\''
                pointing_pol_sub_limited_series = np.array(datafr.query(query_string)['pol_sub']/datafr.query(query_string)['pol_sub_stdv'])
                meas_count_limited = len(pointing_pol_sub_limited_series)
                abs_mean_limited = np.sqrt(meas_count/folded_var)*(np.mean(np.abs(pointing_pol_sub_limited_series))-folded_mean)
                per_pointing_stats[list_title][freq_range][pointing]['abs_mean_limited'] = abs_mean_limited

                query_string = f'{'freq_range'} == \'{freq_range}\'&{'pointing'} == {pointing}& SSINS_flagged == False & source_list == \'{list_title}\''
                pointing_pol_sub_flag_series = np.array(datafr.query(query_string)['pol_sub_flagged'])
                flagged_count = np.sum(pointing_pol_sub_flag_series)
                total_count = len(pointing_pol_sub_flag_series )
                per_pointing_stats[list_title][freq_range][pointing]['flagged_count'] = flagged_count
                per_pointing_stats[list_title][freq_range][pointing]['total_count'] = total_count
        
    return per_pointing_stats
    ###################################


def create_plots(
    list_file,
    array_dict,
    input_directory,
    processed_data_directory,
    time_dim,
    freq_dim,
    shape_dict,
    sky_field_list_association,
    pointing_info_dict,
    pol_bidict,
    initial_freq_flags,
    suffix_add,
    prelim_mode=True,
    threshold=None,
    estimated_pol_sub_stdv=None,
    per_pointing_stats=None,
    output_directory=None,
    pre_flag_on_SSINS_masks=True,
    abs_mean_per_obs_threshold = 5,
    abs_mean_per_pt_threshold = 5,
    plot_instructions=None,
    display_pointing_changes=True,
    split_on_pointings=False,
    allowed_missing_fraction=0.1,
    display_stats=True,
    restricted_list=None,
    raster_num_labels=False,
    show=False,
    ssins_sig_thresh=None
):
    #Creates a long, detailed plot showing much of the data calculated in functions above.
    #This function could use more documentation
    sorted_freq_ranges,sorted_freq_mins = freq_range_sort(shape_dict)
    if prelim_mode==False:
        if threshold is None:
            raise Exception('When not run in preliminary mode create_plots expects an argument for threshold.')
        if estimated_pol_sub_stdv is None:
            raise Exception('When not run in preliminary mode create_plots expects an argument for estimated_pol_sub_stdv.')
            
    if per_pointing_stats is None:
        display_stats=False
        
    title = title_gen(time_dim,freq_dim)
    
    list_title = list_file.split('/')[-1].split('.')[0]
    
    if output_directory is None:
        output_directory = os.path.join(processed_data_directory,list_title)
        
    if restricted_list is not None:
        allowed_missing_fraction=1

    suffix_dict = {'EAVILS':'EAVILS_data.h5','SSINS_data':'SSINS_data.h5','divisor_storage_array':'divisor_storage.npy'}
    suffix_dict = {key:f'{suffix_add}_{suffix_dict[key]}' for key in suffix_dict.keys() }


    filename_dict = get_filenames(
        input_directory,
        suffix_dict=suffix_dict,
        list_or_file=list_file,
        allowed_missing_fraction=allowed_missing_fraction
    )
  
    sky_field_dict = {}
    source_list_dict = {}
    
    

    sky_field = sky_field_list_association[list_title]
    
    plot_arrays_dict = {}
    
    
    obs_tag_list = list(filename_dict.keys())
    obs_tag_list = sorted(obs_tag_list)
    if not restricted_list is None:
        obs_tag_list = [obs_tag for obs_tag in obs_tag_list if obs_tag in restricted_list]
        restricted_tag = 'restricted'
    else:
        restricted_tag=''
    
    # Just pulling telescope name, freq_array and typical time dimensions for plotting
    for obs_tag in filename_dict.keys():
        eavils = wtrf.EAVILS(filename_dict[obs_tag]['EAVILS'])
        freq_array = eavils.freq_array
        time_buffer = 24*3600*(eavils.time_array[-1] - eavils.time_array[0])
        integration_time = 24*3600*(eavils.time_array[1] - eavils.time_array[0])
        instrument_name = eavils.telescope.instrument
        break

    
  
    
    xticks = eavils_utils.freq_ind_finder(freq_array,sorted_freq_mins)
    
    
    xticklabels = [
        "%.0f" % (freq_array[tick] * 10 ** (-6)) for tick in xticks
    ]
    
    
    if plot_instructions is None:
        if prelim_mode:
            plot_instructions =  [('EAVILS','XX','raster',1), 
                                  ('SSINS','XX','raster',1),
                                  ('EAVILS','YY','raster',1),
                                  ('SSINS','YY','raster',1),
                                  ('SSINS_flags','XX','raster',1)
                                 ]
        else:
            plot_instructions =  [
                                  ('variance','XX','raster',1),
                                  ('SSINS','XX','raster',1),
                                  ('EAVILS','XX','raster',1), 
                                  ('pol_sub','XX-YY','line_plot',2),
                                  ('SSINS_flags','XX','raster',1),
                                  ('EAVILS','YY','raster',1),
                                  ('SSINS','YY','raster',1),
                                  ('variance','YY','raster',1)
                                 ]
        
        print(f'Using default plot_instructions:{plot_instructions}')

    
    col_width_multipliers = [entry[3] for entry in  plot_instructions]
                  
    

    color_dict = {'EAVILS':'coolwarm', 'SSINS':'coolwarm', 
                   'variance':'coolwarm',
                  'SSINS_flags':'Greens','pol_sub':'coolwarm'}
    
    vlims_dict = {'EAVILS':(-5,5), 'SSINS':(-5,5), 
                    'variance':(0,2*.95*1.4),
                  'SSINS_flags':(0,1),'pol_sub':(-5,5)}
    
    if instrument_name=='MWA':
        pointing_change_dict={}

    obs_tag_list_split_dict = {}
    for obs_tag in obs_tag_list:
        pointing = pointing_info_dict[obs_tag]['pointing']
        try:
            obs_tag_list_split_dict[pointing].append(obs_tag)
        except KeyError:
            obs_tag_list_split_dict[pointing]=[obs_tag]

    obs_tag_list_split = []
    for pointing,sub_list in obs_tag_list_split_dict.items():
        obs_tag_list_split.append(sub_list)
        pointing_change_dict[pointing] = min(sub_list)
        
    if split_on_pointings:
        obs_tag_list_collection = obs_tag_list_split
    else:
        obs_tag_list_collection = [obs_tag_list]

    
    for obs_tag_sub_list in obs_tag_list_collection:



        #This chunk creates a list of positions in figure coordinates determining where to plot each subfigure based on obs_tag number
        if instrument_name=='MWA':
            positions = [int(obs_tag)-int(obs_tag_sub_list[0]) for obs_tag in obs_tag_sub_list]
        full_vertical_length = positions[-1]+time_buffer
        positions.append(full_vertical_length)
        positions = 1-np.array(positions)/full_vertical_length
        
        goal_aspect = .65
        
        row_count = (int(obs_tag_sub_list[-1])-int(obs_tag_sub_list[0]))/time_buffer

        col_count = sum(col_width_multipliers)
        size_factor=2
        fig_height = row_count*size_factor
        fig_width = col_count*size_factor/goal_aspect
    
        fig = plt.figure(
            figsize=(
                fig_width,
                fig_height),dpi=300)
        
        fig_width, fig_height = fig.get_size_inches()
        title_y = 1 + (0.7 / fig_height) 


        column_width = 1/col_count
        
        prev_pointing = -1 # setting to an arbitrary number that will never be an obs_tag
        
        for ind, obs_tag in enumerate(obs_tag_sub_list):
            
            current_pointing = pointing_info_dict[obs_tag]['pointing']
            if current_pointing!=prev_pointing:
                print(f'pointing = {current_pointing}')
                pointing_change_bool=True
            else:
                pointing_change_bool=False
                
            print(obs_tag)
            
            current_arrays_dict = {}
            

            # This block reads in data from files and sets up data objects
            eavils = wtrf.EAVILS(filename_dict[obs_tag]['EAVILS'])
            
                
            divisor_storage_array = np.load( filename_dict[obs_tag]['divisor_storage_array'])
            ins = INS(filename_dict[obs_tag]['SSINS_data'])
            current_arrays_dict['SSINS'] = ins.metric_ms
            flagged_ins = run_ssins(
                ssins_filename=filename_dict[obs_tag]['SSINS_data'],
                initial_freq_flags=initial_freq_flags,
                shape_dict=shape_dict,
                ssins_sig_thresh=ssins_sig_thresh
            )

            current_arrays_dict['SSINS_flags'] = flagged_ins.mask_to_flags()
            
            if pre_flag_on_SSINS_masks:
                eavils.build_div_and_spectrum(divisor_storage_array,ssins_flags = ins.mask_to_flags()) 
            else:
                eavils.build_div_and_spectrum(divisor_storage_array)

        
            current_arrays_dict['EAVILS'] = eavils.spectrum
            

        
            for stat_type in array_dict.keys():
                current_arrays_dict[stat_type] = array_dict[stat_type][obs_tag]
            

            
            current_arrays_dict['variance'] = np.ma.masked_array(data=current_arrays_dict['variance'],mask=current_arrays_dict['reshaped_SSINS_mask'])
    
    
            pol_sub = array_dict['pol_sub'][obs_tag]
            current_arrays_dict['pol_sub'] = np.ma.masked_array(data=pol_sub,mask=current_arrays_dict['reshaped_SSINS_mask'])
            
            plot_arrays_dict[obs_tag] = current_arrays_dict
            
            col_ind = 0
            for data_title, pol,plot_type,col_width_multiplier in plot_instructions:
                
                current_array = current_arrays_dict[data_title] 
                Ntime_blocks,Nfreq_ranges,Npols = current_array.shape
                if data_title in ['variance','pol_sub']:
                    blocked_bool = True
                    
                elif data_title in ['SSINS','EAVILS','SSINS_flags']:
                    blocked_bool = False
            
                else:
                    raise Exception(f'Unrecognized data title{data_title}')    
        
        
                    
                if blocked_bool:
                    height = size_factor*Ntime_blocks*time_dim/full_vertical_length
                else:
                    height = size_factor*Ntime_blocks/full_vertical_length
                    if data_title!='SSINS_flags':
    
                        initial_flags_extended = initial_freq_flags[np.newaxis,:,np.newaxis]
                        initial_flags_extended = initial_flags_extended*np.ones(current_array.shape)
                        current_array = np.ma.array(
                            current_array,
                            mask=initial_flags_extended
                            )
        
        
                top = positions[ind]
                
                bottom = top - height  # convert top position to bottom for `add_axes`
                
                if col_ind==0:
                    width = height
                if pol in pol_bidict.inverse.keys():
                    pol_ind = pol_bidict.inverse[pol]
                else:
                    polA, polB, = pol.split('-')
                    pol_ind = pol_bidict.inverse[polA]
                
                ax = fig.add_axes([column_width*np.sum(col_width_multipliers[:col_ind]), bottom, column_width*col_width_multiplier, height])  # [left, bottom, width, height]
                default_aspect = Ntime_blocks/Nfreq_ranges
                aspect = goal_aspect/default_aspect
                
                if type(color_dict[data_title])==str:
                    if 'use' in color_dict[data_title]:
                        color_data_title = color_dict[data_title].split('_')[-1]
                    else:
                        color_data_title = data_title
                else:
                    color_data_title = data_title
                    
                cmap = color_dict[color_data_title]
                
                vmin,vmax = vlims_dict[color_data_title]
                if color_data_title==data_title:
                    color_array = deepcopy(current_array)
                else:
                    color_array = current_arrays_dict[color_data_title] 
                    
                #Deals with infs in color array
                color_array[color_array>=vmax]=vmax
                color_array[color_array<=vmin]=vmin
                if blocked_bool:
                    color_array = color_array.data
                if plot_type=='raster':
                    ax.imshow(color_array[:,:,pol_ind],cmap=cmap,vmin=vmin,vmax=vmax,aspect=aspect)
                    if blocked_bool and raster_num_labels:
                        for (i, j), z in np.ndenumerate(current_array[:,:,pol_ind]):
                            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',size=6*size_factor)
    
                
                elif plot_type=='line_plot':
                    max_range = 7.5
                    offsets=[]
                    
                    for f_range_ind,freq_range in enumerate(sorted_freq_ranges):
                        
                        if data_title=='pol_sub':
                            values = ((1/estimated_pol_sub_stdv[sky_field][freq_range]))*current_array[:,f_range_ind,pol_ind]
                            values = np.ma.masked_array(data=values,mask=current_array[:,f_range_ind,pol_ind].mask)
                            
                        else:
                            raise Exception(f'{data_title} incompatible with {plot_type}')

                        
                        


                        plot_values=values.data
                        plot_values = np.append(plot_values, plot_values[-1])
                                                
                        offset = f_range_ind*max_range*2
                        offsets.append(offset)
                        t_spots = np.arange(Ntime_blocks+1,0,-1)
                        
                        ax.axvline(x=offset-threshold,color='black',linewidth=.5,linestyle='--')
                        ax.axvline(x=offset,linewidth=.35,color='black')
                        ax.axvline(x=offset+threshold,color='black',linewidth=.5,linestyle='--')
                        
                        
                        line, = ax.plot(-plot_values+offset,t_spots,linewidth=2, drawstyle="steps")
                        color = line.get_color()
                        ax.fill_betweenx(t_spots, offset, -plot_values+offset, 
                            step="post",facecolor=color, alpha=0.15)
                        
                        
                        '''shading_selection = np.abs(plot_values)>=threshold
                        ax.fill_betweenx(t_spots, offset, -plot_values+offset, 
                            step="post",facecolor=color, alpha=0.6,where=shading_selection)'''

                        for i in range(len(values)):
                            if np.abs(values.data[i]) > threshold:
                                ax.fill_betweenx(
                                    [t_spots[i], t_spots[i+1]],        
                                    [-plot_values[i]+offset, -plot_values[i]+offset],          
                                    [offset, offset],                         
                                    step='post',
                                    facecolor=color, 
                                    alpha=0.4
                                )
                                
                        for i in range(len(values)):

                            if values.mask[i]:
                                ax.fill_betweenx(
                                    [t_spots[i], t_spots[i+1]],        
                                    [-threshold+offset, -threshold+offset],          
                                    [threshold+offset, threshold+offset],                         
                                    step='post',
                                    facecolor='gray', 
                                    alpha=0.5
                                )

                        
                        
                        meas_count = np.sum(~values.mask)
                        folded_mean = np.sqrt(2/np.pi)
                        folded_var = 1-2/np.pi
                        abs_mean = np.sqrt(meas_count/folded_var)*(np.mean(np.abs(values))-folded_mean)
                        height_adjustment = .08
                        if display_stats:
                            text_size = 7
                            
                            
        
                            if abs_mean>=abs_mean_per_obs_threshold:
                                obs_tag_text_box_color = 'red'
                            else:
                                obs_tag_text_box_color = 'white'
        
                            
                            ax.text(offset,0,
                                        ''+'{:0.1f}'.format(abs_mean), ha='center', va='bottom',size=text_size, transform=ax.get_xaxis_transform(),bbox=dict(
                                                alpha=.6, color=obs_tag_text_box_color, mutation_aspect=0.5
                                            ))          
                                                
                            text_height = 1 - height_adjustment
                            
                            if pointing_change_bool:
        
                                abs_mean_value_per_pt = per_pointing_stats[list_title][freq_range][current_pointing]['abs_mean']
                                abs_mean_value_per_pt_limited = per_pointing_stats[list_title][freq_range][current_pointing]['abs_mean_limited']
                                flag_count = per_pointing_stats[list_title][freq_range][current_pointing]['flagged_count']
                                total_count = per_pointing_stats[list_title][freq_range][current_pointing]['total_count']
        
                                
                                if abs_mean_value_per_pt>=abs_mean_per_pt_threshold:
                                    text_box_color = 'red'
                                else:
                                    text_box_color = 'white'
            
                                if abs_mean_value_per_pt_limited>=abs_mean_per_pt_threshold:
                                    text_box_color_limited = 'red'
                                else:
                                    text_box_color_limited = 'white'    
                                    
                                
                                
                                ax.text(offset,text_height,
                                        f'frc: {flag_count}/{total_count}', ha='center', va='bottom',size=text_size, transform=ax.get_xaxis_transform(),bbox=dict(
                                                alpha=.6, color='white', mutation_aspect=0.5
                                            ))
                                ax.text(offset,text_height-height_adjustment,
                                        'abs: '+'{:0.1f}'.format(abs_mean_value_per_pt), ha='center', va='bottom',size=text_size, transform=ax.get_xaxis_transform(),bbox=dict(
                                                alpha=.6, color=text_box_color, mutation_aspect=0.5
                                            ))
                                ax.text(offset,text_height-2*height_adjustment,
                                        'abs_l: '+'{:0.1f}'.format(abs_mean_value_per_pt_limited), ha='center', va='bottom',size=text_size, transform=ax.get_xaxis_transform(),bbox=dict(
                                                alpha=.6, color=text_box_color_limited, mutation_aspect=0.5
                                            ))
                        
                    ax.set_xlim(min(offsets)-max_range,max(offsets)+max_range)
                    ax.set_ylim(min(t_spots),max(t_spots))
                else:
                    raise Exception(f'{plot_type} not recognized')
                
                
    
                if obs_tag == obs_tag_sub_list[-1]:
                    if plot_type=='raster':
                        if blocked_bool:
                            ax.set_xticks(np.arange(len(sorted_freq_ranges)), sorted_freq_ranges,rotation=90,size=10*size_factor)
                        else:
                            
                            ax.set_xticks(xticks )
            
                            ax.set_xlabel("freq (MHz)", fontsize=10 * size_factor)
                            ax.set_xticklabels(xticklabels, fontsize=8 * size_factor)
    
                    elif plot_type=='line_plot':
                        ax.set_xticks(offsets, sorted_freq_ranges,size=16*size_factor)
                else:
                    ax.set_xticks([])
        
                if col_ind==0:
                    if pointing_change_bool and display_pointing_changes:
                        
                        fig.add_artist(
                            lines.Line2D(
                                [0, 1],
                                [top, top],
                                color="magenta",
                            )
                        )
                        mini_text = (
                            f"PTNG:{round(pointing_info_dict[obs_tag]['pointing'],2)},\n"
                            f"RA:{round(pointing_info_dict[obs_tag]['ra'],2)},\n"
                            f"DEC:{round(pointing_info_dict[obs_tag]['dec'],2)}\n"
                            f"ALT:{round(pointing_info_dict[obs_tag]['alt'],2)},\n"
                            f"AZ:{round(pointing_info_dict[obs_tag]['az'],2)}"
                        )
                        ax.text(
                            1.05,
                            top,
                            mini_text,
                            transform=fig.transFigure,
                            fontsize=6 * size_factor,
                            verticalalignment="top",
                            bbox=dict(alpha=0.1, color="magenta"),
                        )
                if obs_tag == obs_tag_sub_list[0]:
                    if data_title=='pol_sub':
                        data_nice_title='Pol. sub.'
                    else:
                        data_nice_title=data_title
                    ax.set_title(f'{data_nice_title}, {pol}',fontsize=14*size_factor)
                if col_ind==0:
                    ax.set_yticks([0])
                    ax.set_yticklabels([obs_tag], fontsize=7.5 * size_factor)
                    ax.tick_params(axis="y", labelrotation=90)
    
                else:
                    ax.set_yticks([])
                
                col_ind+=1
                
            prev_pointing = current_pointing


        if split_on_pointings:
            pointing_tag=current_pointing
        else:
            pointing_tag='all'
        
        plt.suptitle(f'{list_title}, pointing: {pointing_tag}',y=title_y,fontsize=10*size_factor)

        pdf_filename = f'{list_title}_{title}{restricted_tag}_ptng_{pointing_tag}_EAVILS.pdf'
        pdf_filename = os.path.join(output_directory,pdf_filename)
        print(f'saving {pdf_filename}')
        plt.savefig(pdf_filename, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()
    
    
    
    
    
    



def line_plots(list_file,array_dict,processed_data_directory,time_dim,freq_dim,shape_dict,sky_field_list_association,pointing_info_dict,pol_bidict,threshold=None,estimated_pol_sub_stdv=None,per_pointing_stats=None, output_directory=None, pre_flag_on_SSINS_masks=True,abs_mean_per_obs_id_threshold = 5, abs_mean_per_pt_threshold = 5,integration_time=2,plot_instructions=None,display_pointing_changes=True,split_on_pointings=False,allowed_missing_fraction=0.1,display_stats=True,restricted_list=None, line_plot_freq_ranges = None,raster_num_labels=False,show=False):

    if line_plot_freq_ranges is not None:
        time_length = time_dim*integration_time
    
        fig,axes = plt.subplots(2,1,dpi=400,figsize=(12,4))
        plotting_pols = ['XX','YY']
        stretch_factor = 10**2
        ax = axes[0]
    
        for pol in plotting_pols:
            for freq_range_ind,freq_range in enumerate(sorted_freq_ranges):
        
                if freq_range not in line_plot_freq_ranges:
                    continue
            
    
                pol_ind=pol_bidict.inverse[pol]
                plot_list = []
                time_list = []
                for obs_id in obs_id_list:
                    
                    values = plot_arrays_dict[obs_id]['variance'][:,freq_range_ind,pol_ind]
                    
                    plot_list+=list(values)
                    time_list+=list(np.arange(int(obs_id),int(obs_id)+time_length*len(values),time_length))
        
    
                
                line, = ax.plot(time_list,plot_list,linewidth=.5,label=f'{freq_range}, {pol}')
                color = line.get_color()
    
        ax.set_xlabel("OBS ID (GPS seconds)", fontsize=10 )
        ax.ticklabel_format(useOffset=False, style='plain')
        ax.set_ylabel("Variance", fontsize=10 )
        ax.set_aspect(200)
        ymax = 5
        ymin = 0
        yrange = ymax - ymin
        ax.set_aspect(stretch_factor*20/yrange)
        ax.set_ylim(ymin,ymax)
        ax.legend(loc='upper left', prop={'size': 7},framealpha=0.5)
    
        for pointing,p_c_obs in pointing_change_dict.items():
            ax.axvline((int(p_c_obs)), color='magenta', linewidth=1,alpha=.7, linestyle="dotted")
            ax.text(int(p_c_obs)+4*60, 0, f'ptg: {pointing}', ha='left', va='bottom', transform=ax.get_xaxis_transform(),fontsize=8)
        
        ax = axes[1]
        for freq_range_ind,freq_range in enumerate(sorted_freq_ranges):
            if freq_range not in line_plot_freq_ranges:
                continue
            plot_list = []
            time_list = []
            for obs_id in obs_id_list:
                
                values = ((1/estimated_pol_sub_stdv[sky_field][freq_range]))*plot_arrays_dict[obs_id]['pol_sub'][:,freq_range_ind,0]
                
                plot_list+=list(values)
                time_list+=list(np.arange(int(obs_id),int(obs_id)+time_length*len(values),time_length))
                
            
            ax.axhline(y=-threshold,color='black',linewidth=.5,linestyle='--')
            ax.axhline(y=0,linewidth=.35,color='black')
            ax.axhline(y=threshold,color='black',linewidth=.5,linestyle='--')
            line, = ax.plot(time_list,plot_list,linewidth=.5,label=freq_range)
            color = line.get_color()
            ax.fill_between(time_list, 0, plot_list, 
                facecolor=color, alpha=0.3)
            
            shading_selection = eavils_utils.pad_by(np.abs(plot_list)>=threshold,1)[:-1]
            ax.fill_between(time_list, 0, plot_list, 
                facecolor=color, alpha=0.6,where=shading_selection)
    
        ax.set_xlabel("OBS ID (GPS seconds)", fontsize=10 )
        ax.ticklabel_format(useOffset=False, style='plain')
        ax.set_ylabel("Pol. sub. metric", fontsize=10 )
        ax.set_aspect(200)
        ymax = 10
        ymin = -10
        yrange = ymax - ymin
        ax.set_aspect(stretch_factor*20/yrange)
        ax.set_ylim(ymin,ymax)
        ax.legend(loc='upper left',framealpha=0.5)
    
        for pointing,p_c_obs in pointing_change_dict.items():
            ax.axvline((int(p_c_obs)), color='magenta', linewidth=1,alpha=.7, linestyle="dotted")
            ax.text(int(p_c_obs)+4*60, 0, f'ptg: {pointing}', ha='left', va='bottom', transform=ax.get_xaxis_transform(),fontsize=8)
        
        plt.suptitle(f'{list_title}')
        fig.tight_layout() 
        linePlot_filename = f'{list_title}_{title}{restricted_tag}_linePlot.pdf'
        linePlot_filename = os.path.join(output_directory,linePlot_filename)
        plt.savefig(linePlot_filename,bbox_inches="tight")
        if show:
            plt.show()
        plt.close()



def run_ssins(
    ssins_filename,
    initial_freq_flags,
    shape_dict,
    ssins_sig_thresh,
    tb_aggro=0.4,
    broadcast_streak=True,
    time_broadcast=True,
    freq_broadcast=False
):

    ins = INS(ssins_filename)
    #Initial flags
    ins.metric_array.mask = initial_freq_flags[np.newaxis,:,np.newaxis]*np.ones(ins.metric_array.shape)
    shape_dict_orig = deepcopy(shape_dict)
    if 'subTV' in shape_dict_orig.keys():
        del shape_dict_orig['subTV']
    if ssins_sig_thresh is None:
        ssins_sig_thresh = {shape: 5 for shape in shape_dict_orig}
        ssins_sig_thresh["narrow"] = 5
        ssins_sig_thresh["streak"] = 10
    mf = MF(
        ins.freq_array,
        ssins_sig_thresh,
        shape_dict=shape_dict_orig,
        tb_aggro=tb_aggro,
        broadcast_streak=broadcast_streak,
    )
    mf.apply_match_test(
        ins, event_record=True, time_broadcast=time_broadcast, freq_broadcast=freq_broadcast
            )
    
    return ins

