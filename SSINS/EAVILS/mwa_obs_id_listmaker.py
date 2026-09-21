
import yaml
import numpy as np
import copy
import astropy.time as time
import os
from astropy.io.votable import parse_single_table
import argparse
import pandas as pd

def obs_id_listmaker(super_list_file_or_folder='', output_folder='', check_folder=None, check_suffix=None, merge_nights=False,init_obs_id=None,end_obs_id=None):
    if super_list_file_or_folder[-1]=='/':
        super_list_file_or_folder = super_list_file_or_folder[:-1]
    if os.path.isdir(super_list_file_or_folder):
        all_super_lists = [os.path.join(super_list_file_or_folder,file) for file in os.listdir(super_list_file_or_folder)]
        
    else:
        all_super_lists = [super_list_file_or_folder]
        
    #Check for existing obs_ids processed in a folder:
    existing_obs_ids = []
    if not check_folder is None and not check_suffix is None:
        for file in os.listdir(check_folder):
            
            if '.' in check_suffix:
                splitting_char ='.'
            elif '_' in check_suffix:
                splitting_char ='_'
            else:
                raise Exception('Suffix must have a . or a _ in it for proper usage')
            if file.endswith(check_suffix):
                obs_id = file.split(splitting_char)[0]
                create_error=False
                try:
                    int(obs_id)
                except:
                    create_error=True
                if len(obs_id)!=10:
                    create_error=True
                if create_error==True:
                    raise Exception(f'obs id must be interpretable as a 10 digit integer, {obs_id} is not')
                existing_obs_ids.append(obs_id)
    
    
    sorted_full_list = []
    for super_list_file in all_super_lists:
        list_file_suffix = super_list_file.split('.')[-1]
        if list_file_suffix == 'yml':
            
            with open(super_list_file, 'r') as yaml_file:
                yaml_content = yaml.safe_load(yaml_file)
            sorted_full_list += yaml_content.keys()
        
        elif list_file_suffix == 'xml':
            
            xml_table = parse_single_table(super_list_file)
            sorted_full_list += xml_table.array['obs_id'].data
        elif list_file_suffix == 'txt':
            with open(super_list_file, 'r') as txt_file:
                sorted_full_list += txt_file.read().split('\n')
        elif list_file_suffix == 'tsv' or list_file_suffix == 'csv':
            if list_file_suffix == 'tsv':
                seperator = '\t'
            elif list_file_suffix == 'csv':
                seperator = ','
            df = pd.read_csv ("obsdates_qapass_ewp-2..+2.tsv", sep = seperator)
            obs_tag = ''
            for poss_obs_tag in ['obsid','obs_id','OBSID', 'OBS_ID','Obsid','Obs_id']:
                if poss_obs_tag in df.keys():
                    obs_tag = poss_obs_tag
            if obs_tag=='':
                raise Exception(f'no clear obs_id tag found, {df.keys()} are the tags present')
            sorted_full_list += list(df[obs_tag])
                
                
        else:
            try:
                raise ValueError(f"Unsupported list file suffix: {list_file_suffix}")
            except Exception as error:
                print(error)
            
    sorted_full_list = list(set(sorted_full_list)) #Removes duplicates
    sorted_full_list = sorted(sorted_full_list)

    if not init_obs_id is None or not end_obs_id is None:
        print(f'total number of obs_ids before cutting from initial to end obs_id: {len(sorted_full_list)}')
        if not init_obs_id is None:
            sorted_full_list = [obs_id for obs_id in sorted_full_list if int(obs_id)>=int(init_obs_id)]
        if not end_obs_id is None:
            sorted_full_list = [obs_id for obs_id in sorted_full_list if int(obs_id)<=int(end_obs_id)]
        print(f'total number of obs_ids after cutting from initial to end obs_id: {len(sorted_full_list)}')
        
    if not check_folder is None:
        print(f'total number of obs_ids before cutting already downloaded files: {len(sorted_full_list)}')
    sorted_full_list_copy = copy.deepcopy(sorted_full_list)
    sorted_full_list = []
    for obs_id in sorted_full_list_copy:
        if obs_id == '':
            continue
        if (not str(obs_id) in existing_obs_ids and not int(obs_id) in existing_obs_ids):
            sorted_full_list.append(obs_id)
    
    
    
    
    print(f'total number of obs_ids to generate lists for: {len(sorted_full_list)}')
    if merge_nights==True:
        os.makedirs(output_folder, exist_ok=True)  # Ensure directory exists
        file_name = (super_list_file_or_folder.split('/')[-1]).split('.')[0]
        
        output_file = os.path.join(output_folder, f"{file_name}_obsids.txt")
        with open(output_file, "w") as file:
            for obs_id in sorted_full_list:
                if obs_id == '':
                    continue
                if (not str(obs_id) in existing_obs_ids and not int(obs_id) in existing_obs_ids):
                    
                    file.write(str(obs_id) + '\n')
    
    if merge_nights ==False:
        bookending_pairs = []
        prev_entry = 0
        for entry in sorted_full_list:
            entry = int(entry)
            if entry - prev_entry > 43200:
                end = prev_entry
                if end > 0:
                    bookending_pairs.append((start, end))
                start = entry
            prev_entry = entry
        bookending_pairs.append((start, entry))
    
        for pair in bookending_pairs:
            first_obs_id, last_obs_id = pair
            date_string = time.Time(first_obs_id, scale='utc', format='gps').strftime('%m-%d-%Y')
            os.makedirs(output_folder, exist_ok=True)  # Ensure directory exists
            output_file = os.path.join(output_folder, f"{date_string}_obsids.txt")
            with open(output_file, "w") as file:
                for obs_id in sorted_full_list:
                    if (not str(obs_id) in existing_obs_ids and not int(obs_id) in existing_obs_ids):
                        if int(obs_id) >= first_obs_id and int(obs_id) <= last_obs_id:
                            file.write(str(obs_id) + '\n')

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process super_list_file and output_folder.')
    parser.add_argument('super_list_file_or_folder', type=str, help='Path to the input super list file or folder containing lists (YML or XML).')
    parser.add_argument('output_folder', type=str, help='Path to the output folder to save the generated files.')

    # Optional arguments
    parser.add_argument('-c', '--check_folder', type=str, help='Optional: Check a folder to see if files with the input check_suffix exist. If they do corresponding obs_ids will be excluded from the output.')
    parser.add_argument('-s', '--check_suffix', type=str, help='Optional: Suffix for files to be checked in check_folder.')
    parser.add_argument('-m', '--merge_nights', action='store_true', help='Optional: Merge all nights into one list.')
    parser.add_argument('-i', '--init_obs_id', type=str, help='Optional: first obsid to include, anything with smaller value will be cut')
    parser.add_argument('-e', '--end_obs_id', type=str, help='Optional: last obsid to include, anything with larger value will be cut')
    # Parse arguments
    args = parser.parse_args()
    print('input list:', args.super_list_file_or_folder, 'output folder:', args.output_folder)
    
    # Call the function with parsed arguments
    obs_id_listmaker(args.super_list_file_or_folder, args.output_folder, args.check_folder, args.check_suffix, args.merge_nights,args.init_obs_id,args.end_obs_id)
