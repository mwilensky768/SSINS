#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --time=02:00:00
#SBATCH --account=mwaeor
#SBATCH	--output=SSINS_%j.out
#SBATCH --error=SSINS_%j.err

module load singularity
export SINGULARITY_CACHEDIR=$MYGROUP/singularity_cache

#module use /group/mwa/software/modulefiles
#module load six
#module load pyuvdata
#module load h5py
#module load scipy
#module load matplotlib
#module load numpy
#module load pyyaml

data_dir=/astro/mwaeor/MWA/data

echo JOBID $SLURM_ARRAY_JOB_ID
echo TASKID $SLURM_ARRAY_TASK_ID
echo OBSID $obs


gpufiles=$(ls ${data_dir}/${obs}/*gpubox*.fits)

# Only do things if the outputs don't already exist
if [ ! -e ${outdir}/${obs}_SSINS_match_events.yml ]; then

  echo "Executing python script for ${obs}"
  gpufiles=$(ls ${data_dir}/${obs}/*gpubox*.fits)
  metafile=$(ls ${data_dir}/${obs}/*.metafits)
  filelist=""
  for file in $gpufiles; do filelist="$filelist $file"; done
  command="python3 /group/mwa/software/SSINS/mwaf_writer_update/scripts/MWA_vis_to_SSINS.py -o $obs -d ${outdir} -r -f $filelist $metafile"
  srun --export=all env - $(which singularity) exec -B /astro -B /group -B /group/mwaeor/mwilensky $SINGULARITY_CACHEDIR/python3_SSINS_mwaf_writer_update_10_04_2020.sif /bin/bash -c "$command"
#  srun --export=all -n 1 --cpus-per-task 1 env - $(which singularity) exec $SINGULARITY_CACHEDIR/python_ssins-2020-02-13.sif /bin/bash -c "python3 /group/mwa/software/SSINS/csv_script/Scripts/MWA_vis_to_SSINS.py -f $gpufiles $metafile -o $obs -d ${outdir}"
else
  echo "Output already exists. Skipping this obsid ${obs}."
fi
