import os
import mwa_script
import sys
import time
import pandas as pd

obs_id_list_folder = sys.argv[1]
input_data_folder = sys.argv[2]
output_path = sys.argv[3]

all_obs_list = []

for list_file in os.listdir(obs_id_list_folder):
    if list_file.endswith('.txt'):
        with open(f'{obs_id_list_folder}/{list_file}','r') as f:
            obs_id_list = f.read().split('\n')
        for obs_id in obs_id_list:
            all_obs_list.append(obs_id)
false_list = [False]*len(all_obs_list)
dict_for_df = {'obs_id':all_obs_list,'ready':false_list,'complete':false_list,'error':false_list}
df = pd.DataFrame(dict_for_df)

print_waiting_bool=True

while True:
            
    for obs_id in df['obs_id']:

        try:
            
            # Get the modification time (in seconds since the epoch)
            modification_time = os.path.getmtime(f'{input_data_folder}/{obs_id}.uvfits')
            
            # Get the current time (in seconds since the epoch)
            current_time = time.time()
            
            # Calculate the elapsed time
            elapsed_time = current_time - modification_time
            if elapsed_time>120:
                
                
                df.loc[df[df['obs_id']==obs_id].index,'ready']=True

        except FileNotFoundError:
            pass

            
    for obs_id in df.query('ready==True and complete==False')['obs_id']:

        try:
            mwa_script.vis_plotting(
                obs_id,
                input_data_folder=input_data_folder,
                output_path=output_path,
                ss=None,
                output_check=True,
                skip_autos=True,
                extension='uvfits'
            )
            
            df.loc[df[df['obs_id']==obs_id].index,'complete']=True
            
        except Exception as error:
            print(f'Error for {obs_id}: {error}')
            df.loc[df[df['obs_id']==obs_id].index,'complete']=True
            df.loc[df[df['obs_id']==obs_id].index,'error']=True

    if len(df.query('ready==True and complete==False'))==0:
        if print_waiting_bool:
            print('waiting for ready observations')
            print_waiting_bool=False
    else:
        print_waiting_bool=True
