#!/bin/bash

#SBATCH --job-name="MRTA_dataset_generation"
#SBATCH --time=01:00:00
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=education-me-msc-ro

module load 2023r1
module load miniconda3

conda activate /scratch/jbichler/generate_mrta_datasets/dataset_generation_env
srun python /scratch/jbichler/generate_mrta_datasets/code_for_delft_blue/generate_dataset.py 
conda deactivate
