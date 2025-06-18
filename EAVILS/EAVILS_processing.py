import sys
sys.path.insert(1, '/Users/elillesk/repos/SSINS/EAVILS/')
import vis_plotting as vis_plt
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import my_utils
import scipy
import os
import yaml
import copy
import pandas as pd
import bidict

import astropy.io.fits as fits
from astropy.coordinates import Angle
from astropy import units as u



#This function simply sorts the frequency ranges defined in a shape_dict to be ordered in order of their minimum frequency. Should be used in all instances where iterating through frequency ranges is necessary.
def freq_range_sort(shape_dict):
    sorted_freq_ranges = []
    sorted_freq_mins = [shape_dict[freq_range][0] for freq_range in shape_dict.keys()]
    sorted_freq_mins = sorted(sorted_freq_mins)
    for freq_min in sorted_freq_mins:
        for freq_range in shape_dict.keys():
            if shape_dict[freq_range][0] == freq_min:
                sorted_freq_ranges.append(freq_range)
    return sorted_freq_ranges,sorted_freq_mins


#Takes an input directory and identifies all files which have end with the given suffix. 
#Expects files to be in a <obs_id>_<suffix> format. 
#Compares obs_id of these files to a .txt list file with \n as the seperator
#Returns a bidict that can translate between obs_id and file name.
def get_filenames(input_directory,list_file,suffix='spectra_data_cross.h5'):

    with open(list_file,'r') as file:
        obs_id_check_list = file.read().split('\n')
    full_file_list = os.listdir(input_directory)
    full_file_list = [entry for entry in full_file_list if suffix in entry]
    full_file_list = sorted(full_file_list)
    filename_dict = {}
 
            
    for filename in full_file_list:
        obs_id = filename.split('_')[0]
        
        if not obs_id in obs_id_check_list:
            continue
        
    
        
        
        filename_dict[obs_id] = filename
    return bidict.bidict(filename_dict)

#Generates the 'title' which references the number of times and frequencies in a coarse-grain block
def title_gen(time_dim,freq_dim):
    return f'Tdim{time_dim}_Fdim{freq_dim}'


#The primary function for processing raw data (individual obs_id EAVILS h5 files) into a larger set. Coarse grains the data and calculates the frequency across channels. Should be run seperately for each combinations of night and sky field, each of which should be associated with a seperate list_file

#list_file is \n seperated text file with a list of obs_ids associated with a particular night and sky field observation
#input_directory defines where to find the EAVILS .h5 files with the suffix given. They will be compared to obs_ids in the list_file
#output_directory defines where to output the processed data.
#time_dim defines the times coarse-grained across. However, if the time dimension of a given spectrum doesn't divide evenly by the time_dim the remainder is averaged over. In other words, if there are 52 time indices, and the time_dim is 8, there will be 6 blocks of dimension 8 to be averaged over, and one remainder block of dimension 4 which will be averaged over.
#freq_dim defines the frequencies to be coarse-grained across. As it stands, the coarse-graining skips over any permanently flagged channels (such as the coarse-band lines of the MWA). Works similarly to the time_dim but is contained within a frequency channel defined by shape_dict
#pol_bidict just defines the relationship between the names of the polarizations and their indices, (e.g. 'XX' is associated with an index of 0)
#shape_dict defines the possible channels for RFI. In the MWA context this equates to digital TV channels. It's also helpful to include a "clean" channel, where we don't expect digital TV RFI, which can serve as a control for the other channels. However, it's important to note that it's possible to have other types of RFI in this channel, such as narrow-band.
#permanent_flags should be a frequency mask, which has True for flagged channels and False for unflagged channels. This should have the same dimensions as 
#bad_first_time is an option to drop the first time for all arrays, in case of bad data on the first time that was not previously accounted for
#pre_flag_on_SSINS_masks uses the previously generated SSINS mask for each EAVILS spectrum to exclude that data from calculations of the mean and estimated standard deviation. It assumes the SSINS masks have been propagated in polarization which is generally the case.
#clobber overwrites existing processed h5 files where they exist if True.
#suffix determines which suffix the EAVILS h5 files are expected to have. This shouldn't be changed unless previous steps in the pipeline have been changed

def process_data(list_file,input_directory,output_directory,time_dim,freq_dim,pol_bidict,shape_dict,permanent_flags,bad_first_time=True,pre_flag_on_SSINS_masks=True,clobber=False,suffix='spectra_data_cross.h5',return_output=False):
    #This chunk sets up several variables based on inputs
    list_title = list_file.split('/')[-1].split('.')[0]
    output_sub_directory = os.path.join(output_directory,list_title)
    title = title_gen(time_dim,freq_dim)
    h5_name = f'{title}_EAVILS_processed.h5'
    h5_name = os.path.join(output_sub_directory,h5_name)    
    if os.path.exists(h5_name):
        print(f'Output file {h5_name} already exists')
        if clobber:
            print('Overwriting existing file.')
        else:
            print('Exiting function. Change clobber to True to overwrite existing file.')
            return

    sorted_freq_ranges,sorted_freq_mins = freq_range_sort(shape_dict)

    filename_bidict = get_filenames(input_directory,list_file,suffix=suffix)
    first_iteration_bool=True
    per_channel_abs_means_dict = {'times':[]}
    for freq_range in sorted_freq_ranges:
        per_channel_abs_means_dict[freq_range] = {}
        for pol in pol_bidict.inverse.keys():
            per_channel_abs_means_dict[freq_range][pol] = []
    
    #This is the main dictionary that will be saved out to the processed h5 file
    output_data={}
    #This loop iterates through the obs_id and filenames from the list_file present in the input_directory
    for obs_id,filename in filename_bidict.items():
        
    



    
        with h5py.File(os.path.join(input_directory,filename), "r") as hf:
    
            
            freq_array = np.array(hf['blmean'].attrs['freq_array'])
            #Establishes the frequency information used throughout the funbction
            if  first_iteration_bool:
                first_filename = filename
                freq_array_default = freq_array
                range_ind_dict = {}
                range_ind_complete_dict = {}
                if len(freq_array_default)!=len(permanent_flags):
                    raise Exception(f'Length of freq_array {len(freq_array_default)} is not equal to length of the permanent_flags array {len(permanent_flags)}')
                for freq_range in sorted_freq_ranges:
                        
                    freq_range_inds_complete = my_utils.freq_ind_finder(freqs=freq_array,ranges=[shape_dict[freq_range]])
                    range_ind_complete_dict[freq_range] = freq_range_inds_complete
                    
                    freq_range_inds = [ind for ind in freq_range_inds_complete if permanent_flags[ind]==False]
                    
                    freq_range_inds = freq_range_inds[1:-1] #This deals with the fact that we don't want to mix up tv channels or their flags (might be easier way but works for now)
                    range_ind_dict[freq_range] = freq_range_inds
            first_iteration_bool=False
            #Ensures that the frequency array for this data are consistent with that found in the first file.
            if np.any(freq_array!=freq_array_default):
                raise Exception(f'freq_array from {filename} does not match the freq_array from the first file processed, {first_filename}. Please ensure that all obs_ids in {list_file} are from the same set of observations.')
            
            if bad_first_time:
                start_time_index=1
            else:
                start_time_index=0

            #Gets EAVILS data and SSINS mask from h5 file.
            extracted_dict = extract_from_h5(hf,pre_flag_on_SSINS_masks=pre_flag_on_SSINS_masks,start_time_index=start_time_index)
            EAVILS = extracted_dict['EAVILS']
            SSINS_mask = extracted_dict['SSINS_mask']
            
            
        
        
        statistics_dict = {}
        #Perform time coarse-graining within each frequency channel:
        
        for freq_range in sorted_freq_ranges:
            
            superpixel_EAVILS=[]
            superpixel_masks=[]
            
            freq_range_inds = range_ind_dict[freq_range]
            EAVILS_sub_array = EAVILS[:,freq_range_inds,:]
            SSINS_mask_sub_array = SSINS_mask[:,freq_range_inds,:]
            for pol_ind,pol in pol_bidict.items():                                           
                          
                
                superpixel_EAVILS.append(coarse_grain(EAVILS_sub_array[:,:,pol_ind],time_dim,freq_dim,return_same_shape=False,scale_by_sqrt_n=True))
                superpixel_masks.append(coarse_grain(SSINS_mask_sub_array[:,:,pol_ind],time_dim,freq_dim,return_same_shape=False,scale_by_sqrt_n=False,treat_as_boolean=True))
    
    
            
            superpixel_EAVILS = np.array(superpixel_EAVILS)
            superpixel_masks = np.array(superpixel_masks)

            #Fixing the order of axes to maintain the time, frequency, polarization convention used throughout
            superpixel_EAVILS = np.moveaxis(superpixel_EAVILS,0,-1)
            superpixel_masks = np.moveaxis(superpixel_masks,0,-1)

            #Puts several quantities into the statistics_dict:
            statistics_dict[freq_range] = {}
            #coarse-grained data
            statistics_dict[freq_range]['pixel_means'] = superpixel_EAVILS
            #coarse-grained SSINS_mask
            statistics_dict[freq_range]['SSINS_mask'] = superpixel_masks
            
            var_calc = np.var(superpixel_EAVILS,axis=1,ddof=1)
            #The per-time chunk, per frequency-channel, per polarization variance across frequency
            statistics_dict[freq_range]['variance'] = var_calc
    
            _, nfreq_chunks, _ = superpixel_EAVILS.shape #Extracting dimensions of the superpixel array
            dof = nfreq_chunks - 1 #Subtracting one from it because we calculated sample variance
            #The "degrees of freedom" parameter which scales the variance in accordance with a chi-squared distribution parameter
            #In essence this is the number of frequencies which the variance has been calculated over, minus one (due to taking a sample variance)
            statistics_dict[freq_range]['df'] = dof
    
        output_data[obs_id] = statistics_dict
    
    
     
    
    
            
    if not os.path.exists(output_sub_directory):
        os.makedirs(output_sub_directory, mode=0o777)     
    
    

    #Saves out processed data to h5 file
    with h5py.File(h5_name, "w") as hf:
                
        for obs_id in output_data.keys():
            statistics_dict = output_data[obs_id]
            
            save_data = [(freq_range,statistics_dict[freq_range]) for freq_range in statistics_dict.keys()]
            group = hf.create_group(obs_id)
            
            for (freq_range, dct) in save_data:
                subgroup = group.create_group(freq_range)
                subgroup.attrs['df'] = dct['df']
                for key in dct.keys():
                    if key!='df':
                        dset = subgroup.create_dataset(key,data=dct[key])
                    
    
    if return_output==True:
        return output_data
    
    
    
        



    
#Returns pointing information for each obs_id in a dictionary format. Right now is specific to the MWA for EOR-0 and EOR-1 fields. Should be made more general
def pointing_identification(
    metafits_folders, obs_id_list
, include_pointing_int=True):
    #Update
    #It would be good if the EAVILS h5 files just saved this information directly
    if not isinstance(obs_id_list[0], str):
        print(
            "Warning: Correcting type of obs_id_list argument of raw_pointing_list to string"
        )
        obs_id_list = [str(obs_id) for obs_id in obs_id_list]
    pointing_info_dict = {}
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
                    #update to be more general (not mwa specific)
                    if include_pointing_int:
                        pointing = my_utils.mwa_pointings(az, alt, tolerance=0.01)
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





#Coarse grains a 2-dimensional array into blocks of specified size. Default behavior is to average. 
#return_same_shape allows you to choose to return the array with average values recast to original shape.
#scale_by_sqrt_n allows you to multiply each average by the number of included data points to scale noise correctly (only applies if treat_as_boolean is False)
#treat_as_boolean is used for a boolean array, if enabled will just return True if any value in a given block is True
def coarse_grain(arr, block_rows, block_cols, return_same_shape=False, scale_by_sqrt_n=False,treat_as_boolean=False):
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
        return block_means

    # Expand and trim to original shape
    expanded = np.repeat(
        np.repeat(block_means, block_rows, axis=0),
        block_cols, axis=1
    )
    return expanded[:rows, :cols]

#Extracts EAVILS spectrum data and SSINS mask from an already opened h5 file. 
#pre_flag_on_SSINS_masks uses the previously generated SSINS mask for each EAVILS spectrum to exclude that data from calculations of the mean and estimated standard deviation. It assumes the SSINS masks have been propagated in polarization which is generally the case.
#start_time_index can trim the array's start time if desired
def extract_from_h5(hf,pre_flag_on_SSINS_masks,start_time_index=0):
    
    SSINS_mask = np.array(hf['mask'][start_time_index:,:,:])

        
    SSINS_mask = vis_plt.pad_by(SSINS_mask,1)

    if pre_flag_on_SSINS_masks:
        mask_array=SSINS_mask
    else:
        mask_array=None
    corrected_stdv_array = vis_plt.EAVILS_variance(
                        np.array(hf['mean_cross_array'][start_time_index:,start_time_index:,:,:]),
                        mask_array=mask_array,
                        output_stdv=True,
                        maintain_time_dimension=True
                    )
    

    
    blmean = np.array(hf['blmean'][start_time_index:,:,:])
    if pre_flag_on_SSINS_masks:
        blmean = np.ma.masked_array(data=blmean,mask=SSINS_mask)
    EAVILS = (blmean.data-np.mean(blmean,axis=0)
             )/(corrected_stdv_array/np.sqrt(hf['blmean'].attrs['Nbls']))
    EAVILS = EAVILS.data
    return {'EAVILS':EAVILS,'SSINS_mask':SSINS_mask}



#Turns relevant data in the processed data h5 files into per-obs_id arrays, calculates the pol_sub values.
#processed_data_directory is the parent directory for the processed data
#list_titles is the set of list_titles associated with each combination of night and field of observation
#time_dim and freq_dim are as described above, size of coarse-graining blocks, here are just important for selecting the correct processed file
#shape_dict gives the frequency channels where we expect our DTV type RFI
#sky_field_association associates list_titles with sky_fields. This is important for getting statistics right down the road.
#pol_subtraction_order gives the order in which polarizations are subtracted. For consistency, the pol_sub arrays are kept in the same shape as the other arrays despite only the first polarization subtraction combination being used. In general, it is only sensible to subtract polarizations with similar underlying statistics from each other, so e.g. EE-NN or EN-NE, which is why we don't include combinations such as (0,2).

#The function returns:
#an array_dict, which contains the following, all given per time block, per frequency channel, and, in the first two cases, per polarization: variance, the SSINS mask (True if any flagged data in block), and the pol_sub values (the difference of the variance between sets of polarization). 
#a dof_ref_dict or degrees of freedom reference dict, just giving the number of blocks where variances is taken over per frequency channel
#a sky_field_dict which associates each obs_id with its sky_field
#a source_list_dict which associates each obs_id with its source_list
###################################
def create_data_arrays(processed_data_directory,list_titles,time_dim,freq_dim,shape_dict,sky_field_list_association,pol_subtraction_order):
    title=title_gen(time_dim,freq_dim)
    sorted_freq_ranges,sorted_freq_mins = freq_range_sort(shape_dict)
    array_dict = {'variance':{},'reshaped_SSINS_mask':{},'pol_sub':{}}
    dof_ref_dict = {}
    
    sky_field_dict = {}
    source_list_dict = {}
    print('creating data arrays')
    first_time_bool=True
    for list_title in list_titles:

        sky_field = sky_field_list_association[list_title]
        print(list_title)
        h5_name = f'{processed_data_directory}/{list_title}/{title}_EAVILS_processed.h5'
        
        with h5py.File(h5_name, "r") as hf:
            obs_id_sub_list = list(hf.keys())

    
            for obs_id in obs_id_sub_list:
                sky_field_dict[obs_id] = sky_field
                source_list_dict[obs_id] = list_title
                #Expected shape should hold for different frequencies
                Ntime_blocks, Npols = hf[obs_id][sorted_freq_ranges[0]]['variance'].shape
                Nfreq_ranges = len(hf[obs_id].keys())
                
                var_plot_array = np.full((Ntime_blocks,Nfreq_ranges,Npols),np.nan)
                sigma_plot_array = np.full((Ntime_blocks,Nfreq_ranges,Npols),np.nan)
                prob_plot_array = np.full((Ntime_blocks,Nfreq_ranges,Npols),np.nan)
                reshaped_mask_array = np.full((Ntime_blocks,Nfreq_ranges,Npols),np.nan)
                
                for f_range_ind, freq_range in enumerate(sorted_freq_ranges):
                    if hf[obs_id][freq_range]['variance'].shape!=(Ntime_blocks,Npols):
                        raise Exception(f'Error: incorrect array shape for {obs_id},{freq_range},{'variance'}; found:{hf[obs_id][freq_range]['variance'].shape}, expected: {(Ntime_blocks,Npols)}' )
                    
                        
                    var_plot_array[:,f_range_ind,:]=np.array(hf[obs_id][freq_range]['variance'])
                    reshaped_mask_array[:,f_range_ind,:]=np.any(np.array(hf[obs_id][freq_range]['SSINS_mask']),axis=1)
                    
                    dof_ref_dict[freq_range]=hf[obs_id][freq_range].attrs['df']

                if first_time_bool:
                    default_dof_ref_dict = copy.deepcopy(dof_ref_dict)
                    first_time_bool=False
                for f_range_ind, freq_range in enumerate(sorted_freq_ranges):
                    if dof_ref_dict[freq_range]!=default_dof_ref_dict[freq_range]:
                        raise Exception(f'dof_ref_dict inconsistent for {obs_id}, {freq_range}. Expected {default_dof_ref_dict[freq_range]}, found {dof_ref_dict[freq_range]} Not designed to handle this kind of discrepancy.')
                    
                pol_sub_array = np.full(var_plot_array.shape,np.nan)
                for pol_ind,(polA_ind,polB_ind) in enumerate(pol_subtraction_order):
                    pol_sub_array[:,:,pol_ind] = var_plot_array[:,:,polA_ind] - var_plot_array[:,:,polB_ind]
                    
                array_dict['variance'][obs_id] = var_plot_array
                array_dict['pol_sub'][obs_id]  = pol_sub_array
                array_dict['reshaped_SSINS_mask'][obs_id] = reshaped_mask_array
    return array_dict, dof_ref_dict,sky_field_dict,source_list_dict
            
    
    ##########################################################################

#pol_bidict associates the polarizations with their indices
#shape_dict gives the frequency channels where we expect our DTV type RFI
#array_dict is the result of the create_array_dict function explicated above
#sky_field_dict and source_list_dict are similarly explicated above
#metafits_folders is used for extracting pointing information. 

#Returns a pandas dataframe which collects the important data for each measurement into a single pandas dataframe. In other words, for each frequency and time, their is a row with all the input data.
def create_data_frame(pol_bidict,shape_dict,array_dict,sky_field_dict,source_list_dict,metafits_folders):
    sorted_freq_ranges,sorted_freq_mins = freq_range_sort(shape_dict)
    obs_id_list = list(array_dict['variance'].keys())
    pointing_info_dict = pointing_identification([metafits_folders],obs_id_list)
    datafr_dict={'obs_id':[],'sky_field':[],'t_block_ind':[],'pointing':[],'pol_sub':[],'SSINS_flagged':[],'pol_sub_flagged':[],'pointing_flagged':[],'source_list':[]}
    for pol in pol_bidict.inverse.keys():
        datafr_dict[f'variance_{pol}']=[]
    for freq_range in sorted_freq_ranges:
        datafr_dict['freq_range']=[]
    
    for obs_id in obs_id_list:
        Ntime_blocks,_,_ = array_dict['variance'][obs_id].shape
        for freq_range_ind, freq_range in enumerate(sorted_freq_ranges):
            sky_field = sky_field_dict[obs_id]
            source_list = source_list_dict[obs_id]
            datafr_dict['obs_id']+=[obs_id for i in range(Ntime_blocks)]
            datafr_dict['sky_field']+=[sky_field for i in range(Ntime_blocks)]
            datafr_dict['source_list']+=[source_list for i in range(Ntime_blocks)]
            datafr_dict['pointing']+=[pointing_info_dict[obs_id]['pointing'] for i in range(Ntime_blocks)]
            datafr_dict['freq_range']+=[freq_range for i in range(Ntime_blocks)]
            datafr_dict['t_block_ind']+=[i for i in range(Ntime_blocks)]
            
            for pol_ind,pol in pol_bidict.items():
                datafr_dict[f'variance_{pol}']+=list(array_dict['variance'][obs_id][:,freq_range_ind,pol_ind])
    
            
            datafr_dict[f'SSINS_flagged']+=list(np.bool_(array_dict['reshaped_SSINS_mask'][obs_id][:,freq_range_ind,0]))
    
            
            pol_sub_mini_list = (array_dict['pol_sub'][obs_id][:,freq_range_ind,0])
            
            datafr_dict['pol_sub']+=list(pol_sub_mini_list)
            datafr_dict['pol_sub_flagged']+=[False for i in range(Ntime_blocks)]
            datafr_dict['pointing_flagged']+=[False for i in range(Ntime_blocks)]
        
        
        
    
    datafr=pd.DataFrame(datafr_dict)
    return datafr


#This adds the column for the estimated pol_sub standard deviation, necessary for scaling thresholds. This must be calculated independently for each sky field and array configuration. Its critical that this is done correctly for expected results.
def add_pol_sub_stdv(datafr,estimated_pol_sub_stdv):
    datafr=copy.deepcopy(datafr)
    datafr['pol_sub_stdv'] = np.nan
    
    for sky_field in list(estimated_pol_sub_stdv.keys()):
        for freq_range in list(estimated_pol_sub_stdv[sky_field].keys()):
    
            selected_inds = datafr.query(f'freq_range == \'{freq_range}\' & sky_field == \'{sky_field}\'').index
            datafr.loc[selected_inds,'pol_sub_stdv'] = estimated_pol_sub_stdv[sky_field][freq_range]
    missing_values = np.sum(pd.isna(datafr['pol_sub_stdv']))
    if missing_values>0:
        raise Exception(f'Missing {missing_values} values. Please ensure that estimated_pol_sub_stdv input contains all sky_fields and freq_ranges included in the datafr')
    return datafr


#This calculates per pointing stats for each list_title, including how many pol_sub measurements crossed the thrshold set, as well as the scaled absolute mean (scaled according to the expected mean and variance for a folded normal distribution, and then the same calculation with threshold crossing measurements removed.
def find_per_pointing_stats(datafr,shape_dict,list_titles,threshold):
    sorted_freq_ranges,sorted_freq_mins = freq_range_sort(shape_dict)
    datafr['pol_sub_flagged'] = np.abs(datafr['pol_sub'])>(datafr['pol_sub_stdv']*threshold)
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

#Creates a long, detailed plot showing much of the data calculated in functions above.
#The inputs have mostly been previously documented, with the exception of:
#time_buffer, which provides 
def create_plots(list_file,array_dict,per_pointing_stats,dof_ref_dict,input_directory,processed_data_directory,time_dim,freq_dim,shape_dict,sky_field_list_association,metafits_folders,pol_bidict,threshold,estimated_pol_sub_stdv,permanent_flags,time_buffer, bad_first_time=True,pre_flag_on_SSINS_masks=True,abs_mean_per_obs_id_threshold = 5, abs_mean_per_pt_threshold = 5,integration_time=2,suffix='spectra_data_cross.h5'):
    title = title_gen(time_dim,freq_dim)
    list_title = list_file.split('/')[-1].split('.')[0]
    
    output_sub_directory = os.path.join(processed_data_directory,list_title)
    filename_bidict = get_filenames(input_directory, list_file,suffix=suffix)
    sorted_freq_ranges,sorted_freq_mins = freq_range_sort(shape_dict)
    


    #obs_id_list = []
    
    sky_field_dict = {}
    source_list_dict = {}
    
    

    sky_field = sky_field_list_association[list_title]
    
    h5_name = f'{processed_data_directory}/{list_title}/{title}_EAVILS_processed.h5'
    plot_arrays_dict = {}
    
    with h5py.File(h5_name, "r") as hf:
        obs_id_list = list(hf.keys())
    obs_id_list = sorted(obs_id_list)
    pointing_info_dict = pointing_identification([metafits_folders],obs_id_list)
    
    #Update: this chunk should instead be saved as an attribute for the processed h5 file
    

    for filename in os.listdir(input_directory):
        if 'spectra_data_cross.h5' in filename and obs_id_list[0] in filename:

            with h5py.File(os.path.join(input_directory,filename), "r") as hf:
                freq_array = hf['blmean'].attrs['freq_array']
    
    #This chunk creates a list of positions in figure coordinates determining where to plot each subfigure based on obs_id number
    positions = [int(obs_id)-int(obs_id_list[0]) for obs_id in obs_id_list]
    full_t_length = positions[-1]+time_buffer
    positions.append(full_t_length)
    positions = 1-np.array(positions)/full_t_length
    
    goal_aspect = .65
    
    
    
    
    xticks = my_utils.freq_ind_finder(freq_array,sorted_freq_mins)
    
    
    xticklabels = [
                        "%.0f" % (freq_array[tick] * 10 ** (-6)) for tick in xticks
                    ]
    
    
    
    
    data_pol_order = [('variance','XX'),
                      ('SSINS','XX'),
                      ('EAVILS','XX'), 
                      ('pol_sub','XX-YY'),
                      ('SSINS mask','XX'),
                      ('EAVILS','YY'),
                      ('SSINS','YY'),
                      ('variance','YY')
                     ]
    
    plot_types = ['raster',
                  'raster',
                  'raster',
                  'line_plot',
                  'raster',
                  'raster',
                  'raster',
                  'raster']
    
    col_width_multipliers = [ 1,
                              1,
                              1,
                              2,
                              1,
                              1,
                              1,
                              1]
                  
    
    #default_cmap = plt.cm.coolwarm
    #default_cmap.set_bad('red')
    color_dict = {'EAVILS':'coolwarm','EAVILS_coarse':'coolwarm', 'SSINS':'coolwarm', 
                  'sigma':'coolwarm', 'probability':'viridis_r', 'variance':'use_sigma',
                  'SSINS mask':'Greens','pol_sub':'coolwarm'}
    
    vlims_dict = {'EAVILS':(-5,5),'EAVILS_coarse':(-5,5), 'SSINS':(-5,5), 
                  'sigma':(-5,5), 'probability':(0,1), 'variance':(0,2),
                  'SSINS mask':(0,1),'pol_sub':(-5,5)}
    
    
        
        
    
    
    row_count = (int(obs_id_list[-1])-int(obs_id_list[0]))/time_buffer
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
    plt.suptitle(f'{list_title}, time averaging={time_dim*integration_time} sec',y=title_y,fontsize=10*size_factor)
    column_width = 1/col_count
    
    prev_pointing = -10**6
    for ind, obs_id in enumerate(obs_id_list):
        
        current_pointing = pointing_info_dict[obs_id]['pointing']
        if current_pointing!=prev_pointing:
            print(f'pointing = {current_pointing}')
            pointing_change_bool=True
        else:
            pointing_change_bool=False
            
        print(obs_id)
        
        current_arrays_dict = {}
        
        filename = filename_bidict[obs_id]
        #with h5py.File(f'/Volumes/Data3/elillesk/data_and_analysis/vis_plt_outputs/nichole_list/h5_files/{obs_id}_spectra_data_cross.h5', "r") as hf:
        
        with h5py.File(os.path.join(input_directory,filename), "r") as hf:
            dataset_names = list(hf.keys())
            #update
            if bad_first_time:
                start_time_index=1
            else:
                start_time_index=0
            freq_array = hf['blmean'].attrs['freq_array']
            
            
    
    
            extracted_dict = extract_from_h5(hf,pre_flag_on_SSINS_masks=pre_flag_on_SSINS_masks,start_time_index=start_time_index)
            EAVILS = extracted_dict['EAVILS']
            SSINS_mask = extracted_dict['SSINS_mask']
            
            
            
    
    
            current_arrays_dict['EAVILS'] = EAVILS  
            
            '''EAVILS_coarse = np.full(EAVILS.shape,np.nan)
            for pol_ind in range(4):
                EAVILS_coarse[:,:,pol_ind] = coarse_grain(EAVILS[:,:,pol_ind],time_dim,1,return_same_shape=True,scale_by_sqrt_n=True)
                                                      
            current_arrays_dict['EAVILS_coarse'] = EAVILS_coarse'''
            current_arrays_dict['SSINS'] = np.array(hf['metric_ms'][start_time_index:,:,:])
            current_arrays_dict['SSINS mask'] = np.array(hf['mask'][start_time_index:,:,:])
            
    
        for stat_type in array_dict.keys():
            current_arrays_dict[stat_type] = array_dict[stat_type][obs_id]
        
    
        #update
        var_array =  array_dict['variance'][obs_id]
                   
        means_array = np.full(var_array.shape,np.nan)
    
    
        current_arrays_dict['probability'] = np.full(var_array.shape,np.nan)
        for f_range_ind, freq_range in enumerate(sorted_freq_ranges):
            dof = dof_ref_dict[freq_range]
            
            current_arrays_dict['probability'][:,f_range_ind,:] =  scipy.stats.chi2.sf(var_array[:,f_range_ind,:]*dof, df=dof)
    
        current_arrays_dict['sigma'] = scipy.stats.norm.isf(current_arrays_dict['probability'])
        
        
        current_arrays_dict['variance'] = np.ma.masked_array(data=current_arrays_dict['variance'],mask=current_arrays_dict['reshaped_SSINS_mask'])


        pol_sub = array_dict['pol_sub'][obs_id]
        current_arrays_dict['pol_sub'] = np.ma.masked_array(data=pol_sub,mask=current_arrays_dict['reshaped_SSINS_mask'])
        
        plot_arrays_dict[obs_id] = current_arrays_dict
        
        col_ind = 0
        for data_pol,plot_type,col_width_multiplier in zip(data_pol_order,plot_types,col_width_multipliers):
            data_title, pol = data_pol
            current_array = current_arrays_dict[data_title] 
            Ntime_blocks,Nfreq_channels,Npols = current_array.shape
            if data_title in ['sigma','variance','probability','pol_sub']:
                blocked_bool = True
                
            elif data_title in ['SSINS','EAVILS','SSINS mask','EAVILS_coarse']:
                blocked_bool = False
        
            else:
                raise Exception(f'Unrecognized data title{data_title}')    
    
    
                
            if blocked_bool:
                height = size_factor*Ntime_blocks*time_dim/full_t_length
            else:
                height = size_factor*Ntime_blocks/full_t_length
                if data_title!='SSINS mask':

                    permanent_flags_extended = permanent_flags[np.newaxis,:,np.newaxis]
                    permanent_flags_extended = permanent_flags_extended*np.ones(current_array.shape)
                    current_array = np.ma.array(
                        current_array,
                        mask=permanent_flags_extended
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
            default_aspect = Ntime_blocks/Nfreq_channels
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
                color_array = copy.deepcopy(current_array)
            else:
                color_array = current_arrays_dict[color_data_title] 
                
            #Deals with infs in color array
            color_array[color_array>=vmax]=vmax
            color_array[color_array<=vmin]=vmin
            if plot_type=='raster':
                ax.imshow(color_array[:,:,pol_ind],cmap=cmap,vmin=vmin,vmax=vmax,aspect=aspect)
                if blocked_bool:
                    for (i, j), z in np.ndenumerate(current_array[:,:,pol_ind]):
                        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',size=1*size_factor)

            
            elif plot_type=='line_plot':
                max_range = 7.5
                offsets=[]
                
                for f_range_ind,freq_range in enumerate(sorted_freq_ranges):
                    offset = f_range_ind*max_range*2
                    offsets.append(offset)
                    t_spots = np.arange(Ntime_blocks,0,-1)
                    
                    ax.axvline(x=offset-threshold,color='black',linewidth=.5,linestyle='--')
                    ax.axvline(x=offset,linewidth=.35,color='black')
                    ax.axvline(x=offset+threshold,color='black',linewidth=.5,linestyle='--')
                    
                    if data_title=='pol_sub':
                        values = ((1/estimated_pol_sub_stdv[sky_field][freq_range]))*current_array[:,f_range_ind,pol_ind]
                        
                    else:
                        raise Exception(f'{data_title} incompatible with {plot_type}')
                    line, = ax.plot(-values+offset,t_spots,linewidth=2)
                    color = line.get_color()
                    ax.fill_betweenx(t_spots, offset, -values+offset, 
                        facecolor=color, alpha=0.3)
                    
                    shading_selection = vis_plt.pad_by(np.abs(values)>=threshold,1)[:-1]
                    ax.fill_betweenx(t_spots, offset, -values+offset, 
                        facecolor=color, alpha=0.6,where=shading_selection)
                    
                    meas_count = np.sum(~values.mask)
                    folded_mean = np.sqrt(2/np.pi)
                    folded_var = 1-2/np.pi
                    abs_mean = np.sqrt(meas_count/folded_var)*(np.mean(np.abs(values))-folded_mean)


                    text_size = 7
                    height_adjustment = .08
                    

                    if abs_mean>=abs_mean_per_obs_id_threshold:
                        obs_id_text_box_color = 'red'
                    else:
                        obs_id_text_box_color = 'white'

                    
                    ax.text(offset,0,
                                ''+'{:0.1f}'.format(abs_mean), ha='center', va='bottom',size=text_size, transform=ax.get_xaxis_transform(),bbox=dict(
                                        alpha=.6, color=obs_id_text_box_color, mutation_aspect=0.5
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
                ax.set_ylim(min(t_spots)-.5,max(t_spots)+.5)
            else:
                raise Exception(f'{plot_type} not recognized')
            
            

            if obs_id == obs_id_list[-1]:
                if plot_type=='raster':
                    if blocked_bool:
                        ax.set_xticks(np.arange(len(sorted_freq_ranges)), sorted_freq_ranges,size=2.5*size_factor)
                    else:
                        
                        ax.set_xticks(xticks )
        
                        ax.set_xlabel("freq (MHz)", fontsize=1 * size_factor)
                        ax.set_xticklabels(xticklabels, fontsize=4 * size_factor)

                elif plot_type=='line_plot':
                    ax.set_xticks(offsets, sorted_freq_ranges,size=2.5*size_factor)
            else:
                ax.set_xticks([])
    
            if col_ind==0:
                if pointing_change_bool:
                    
                    fig.add_artist(
                        lines.Line2D(
                            [0, 1],
                            [top, top],
                            color="magenta",
                        )
                    )
                    mini_text = (
                        f"PTNG:{round(pointing_info_dict[obs_id]['pointing'],2)},\n"
                        f"RA:{round(pointing_info_dict[obs_id]['ra'],2)},\n"
                        f"DEC:{round(pointing_info_dict[obs_id]['dec'],2)}\n"
                        f"ALT:{round(pointing_info_dict[obs_id]['alt'],2)},\n"
                        f"AZ:{round(pointing_info_dict[obs_id]['az'],2)}"
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
            if obs_id == obs_id_list[0]:
                ax.set_title(f'{data_title}, {pol}')
            if col_ind==0:
                ax.set_yticks([0])
                ax.set_yticklabels([obs_id], fontsize=6 * size_factor)
                ax.tick_params(axis="y", labelrotation=90)

            else:
                ax.set_yticks([])
            
            col_ind+=1
            
        prev_pointing = current_pointing
    
    pdf_filename = f'{list_title}_{title}_EAVILS.pdf'
    pdf_filename = os.path.join(output_sub_directory,pdf_filename)
    print(f'saving {pdf_filename}')
    plt.savefig(pdf_filename, bbox_inches="tight")
    
    plt.close()
    
    
    
    
    
    time_length = time_dim*integration_time
    
    fig,ax = plt.subplots(1,1,dpi=400,figsize=(12,4))
    for freq_range_ind,freq_range in enumerate(sorted_freq_ranges):
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
        
        shading_selection = vis_plt.pad_by(np.abs(plot_list)>=threshold,1)[:-1]
        ax.fill_between(time_list, 0, plot_list, 
            facecolor=color, alpha=0.6,where=shading_selection)
        
        ax.set_aspect(200)
        ax.set_ylim(-10,10)
    plt.suptitle(f'Pol_sub rescaled')
    plt.legend()
    linePlot_filename = f'{list_title}_{title}_linePlot.pdf'
    linePlot_filename = os.path.join(output_sub_directory,linePlot_filename)
    plt.savefig(linePlot_filename,bbox_inches="tight")
    plt.close()




    
