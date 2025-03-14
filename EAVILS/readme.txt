Example call for vis_plotting:
vis_plotting.vis_plotting(obs_id=<obs_id>, plot_types=['spectra'], uvfits_folder = <uvfits_folder>, output_path=<output_path>)

Note that a good way to run this code is by making a simple script that iterates throught obs_ids that exist in the specified uvfits_folder.
The vis_plotting code will create h5 files which will be used in the time_extended module.



Example call for time_extended (from command line):
python time_extended.py <h5_files folder> -l <obs_id_list.txt> -m <metafits folder (usually same as uvfits folder above)>
