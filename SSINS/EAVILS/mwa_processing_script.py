import eavils_processing as proc
import eavils_utils

from copy import deepcopy
import os
import yaml

'''output_plot_directory = '/Volumes/Data3/elillesk/data_and_analysis/processed_outputs/2014_plots'
list_file_directory = '2014_stuff/2014_sub_lists/'
metafits_folder = '2014_stuff/2014_metafits/'
processed_data_directory = '2014_stuff/'
pol_sub_stdv_yaml = '2014_stuff/2014_fall_pol_sub_stdv.yml'
waterfall_file_directory = '/Volumes/Data3/elillesk/data_and_analysis/waterfall_data/2014_birli/h5_files/''''

output_plot_directory = '/Volumes/Data3/elillesk/data_and_analysis/processed_outputs/2016_plots'
list_file_directory = 'example_obs_lists'
metafits_folder = '/Volumes/Data4/elillesk/data_and_analysis/uvfits/chisq_obs/'
processed_data_directory = '2014_stuff/'
pol_sub_stdv_yaml = '2014_stuff/2014_fall_pol_sub_stdv.yml'
waterfall_file_directory = '/Volumes/Data3/elillesk/data_and_analysis/waterfall_data/2014_birli/h5_files/'

threshold = 4.62 #Rounding to three decimal places
time_spacing = 120
integration_time = 2
complete_processing = False
###########################################################
#FIX THIS PART
#Establishing which data is in the EOR0 and EOR1 sky fields

EOR0_lists = []
EOR1_lists = []
for file in os.listdir(list_file_directory):
    if 'EOR0' in file:
        EOR0_lists.append(file)
    elif 'EOR1' in file:
        EOR1_lists.append(file)
    


all_lists =  EOR0_lists+EOR1_lists
all_lists = sorted(all_lists)

list_titles = [file.split('.')[0] for file in os.listdir(list_file_directory) if len(file.split('.')[0] )>0]
list_titles = sorted(list_titles)
sky_field_list = ['EOR0']
sky_field_list_association = {list_file.split('.')[0]:'EOR0' for list_file in EOR0_lists}
sky_field_list_association.update({list_file.split('.')[0]:'EOR1' for list_file in EOR1_lists})

###########################################################



#Establishes the order in which polarizations should be subtracted
pol_subtraction_order = {(0,1),(1,0),(2,3),(3,2)}


#Flagging the coarse band lines for the MWA. In general this parameter can be used for flagging coarse-band lines,
#or things like known FM or satellite contaminated channels
initial_freq_flags = eavils_utils.coarse_band_flagging()

#This shape_dict is taken from SSINS, with an additional "subTV" pseudo-channel added as a control "clean channel". 
#This subTV channel is quite simply the frequencies below which the digital TV channels live
shape_dict = eavils_utils.get_shape_dict('MWA_high',add_subTV=True)
#We only want TV channels, so we remove the 'center_packet_loss' frequency type
del shape_dict['center_packet_loss']
print(shape_dict)
#This quite simply sorts our frequency channels based on their minimum frequency
sorted_freq_ranges,sorted_freq_mins = proc.freq_range_sort(shape_dict)
#Establishes the correspondence between polarizations and polarization indices
pol_dict = bidict.bidict({0:'XX',1:'YY',2:'XY',3:'YX'})


time_dim = 8 #Number of integrations averaged across
freq_dim = 1 #Number of frequency channels averaged across

shape_dict_orig = deepcopy(shape_dict)
if 'subTV' in shape_dict_orig.keys():
    del shape_dict_orig['subTV']

ssins_sig_thresh = {shape: 5 for shape in shape_dict_orig}
ssins_sig_thresh["narrow"] = 5
ssins_sig_thresh["streak"] = 10

for list_file in all_lists:
    
    list_title=list_file.split('.txt')[0]
    list_file = f'{list_file_directory}/{list_file}'
    print(list_title, list_file)
    proc.process_data(list_file=list_file,
        input_directory=waterfall_file_directory,
        output_directory=processed_data_directory,
        time_dim=time_dim,
        freq_dim=freq_dim,
        pol_dict=pol_dict,
        shape_dict=shape_dict,
        initial_freq_flags=initial_freq_flags,
        suffix_add='80khz_cross',
        metafits_folder=metafits_folder,
        pre_flag_on_SSINS_masks=True,
        clobber=False, 
        allowed_missing_fraction=.1,
        ssins_sig_thresh=ssins_sig_thresh
    )

if complete_processing == True:
    
    array_dict, sky_field_dict,source_list_dict,combined_pointing_info_dict = proc.create_data_arrays(
        processed_data_directory = processed_data_directory,
        list_titles=list_titles,
        time_dim=time_dim,
        freq_dim=freq_dim,
        shape_dict=shape_dict,
        sky_field_list_association=sky_field_list_association,
        pol_subtraction_order=pol_subtraction_order,
        add_pointing_dict=True
    )
    
    
    
    
    datafr = proc.create_data_frame(
        pol_dict=pol_dict,
        shape_dict=shape_dict,
        array_dict=array_dict,
        sky_field_dict=sky_field_dict,
        source_list_dict=source_list_dict,
        pointing_info_dict=combined_pointing_info_dict
    )
    
    with open(pol_sub_stdv_yaml, 'r') as file:
        estimated_pol_sub_stdv = yaml.safe_load(file) # Use safe_load for security
    
    datafr= proc.add_pol_sub_stdv(datafr,estimated_pol_sub_stdv,threshold)
    
    per_pointing_stats=proc.find_per_pointing_stats(datafr=datafr,shape_dict=shape_dict,list_titles=list_titles,threshold=threshold)
    
    for list_file in all_lists:
        list_file = f'{list_file_directory}/{list_file}'
        proc.create_plots(
            list_file = list_file,
            array_dict = array_dict,
            input_directory = waterfall_file_directory,
            processed_data_directory = processed_data_directory,
            time_dim = time_dim,
            freq_dim = freq_dim,
            shape_dict = shape_dict,
            sky_field_list_association = sky_field_list_association,
            pointing_info_dict = combined_pointing_info_dict,
            pol_dict = pol_dict,
            initial_freq_flags = initial_freq_flags,
            suffix_add = '80khz_cross',
            prelim_mode=False,
            output_directory = output_plot_directory,
            pre_flag_on_SSINS_masks=False,
            display_pointing_changes=False,
            split_on_pointings=True,
            display_stats = False,
            raster_num_labels=False,
            show=False,
            threshold=threshold,
            estimated_pol_sub_stdv=estimated_pol_sub_stdv
        )