import os
import mwa_script
import sys

obs_id_list_folder = sys.argv[1]
input_data_folder = sys.argv[2]
output_path = sys.argv[3]

for list_file in os.listdir(obs_id_list_folder):
    with open(f'{obs_id_list_folder}/{list_file}','r') as f:
        obs_id_list = f.read().split('\n')
    for obs_id in obs_id_list:
        try:
            mwa_script.vis_plotting(
            obs_id,
            input_data_folder=input_data_folder,
            output_path=output_path,
            ss=None,
            output_check=True,
            skip_autos=False,
            extension='uvfits'
        )
            
        except Exception as error:
            print(f'Error making movie/spectra plots: {error}')