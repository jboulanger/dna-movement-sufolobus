#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=12
#SBATCH --time=05:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Slurm script to process file on an HPC cluster using a container

echo "TASK ID : $SLURM_ARRAY_TASK_ID"
git rev-parse --short HEAD
date 

r= # path to data folder
d= # path to results folder

time apptainer exec --nv \
	--writable-tmpfs \
	--bind /cephfs2:/cephfs2,/cephfs:/cephfs,/lmb:/lmb \
	/public/singularity/containers/lightmicroscopy/bioimaging-container/bioimaging.sif \
	/bin/micromamba run -n imaging \
	python dnasufo.py process \
	--root $r\
	--dst $d\
	--index $SLURM_ARRAY_TASK_ID
	
date