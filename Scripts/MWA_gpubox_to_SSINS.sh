#!/bin/bash

module use /group/mwa/software/modulefiles
module load six
module load pyuvdata
module load h5py
module load scipy
module load matplotlib
module load numpy
module load pyyaml

data_dir=/astro/mwaeor/MWA/data

echo JOBID $SLURM_ARRAY_JOB_ID
echo TASKID $SLURM_ARRAY_TASK_ID

obs=$(sed "${SLURM_ARRAY_TASK_ID}q;d" ${obs_file_name})
echo OBSID $obs

gpufiles=$(ls ${data_dir}/${obs}/*gpubox*.fits)

# Only do things if the outputs don't already exist
if [ ! -e ${outdir}/${obs}_SSINS_data.h5 ]; then

  echo "Executing python script for ${obs}"
  gpufiles=$(ls ${data_dir}/${obs}/*gpubox*.fits)
  python MWA_vis_to_SSINS.py -f $gpufiles -o $obs ${uvfits_dir}/${obs}_noflag.uvfits -d ${outdir}
fi
