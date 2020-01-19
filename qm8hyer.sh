#!/bin/bash
# Begin LSF Directives
#BSUB -P med106
#BSUB -W 12:00
#BSUB -nnodes 22 
#BSUB -q killable
#BSUB -J QMhyperopt
#BSUB -o QMhyperopt.%J
#BSUB -e QMhyperopt.%J

module load python/3.6.6-anaconda3-5.3.0
module load gcc/4.8.5
cd /gpfs/alpine/med106/proj-shared/aclyde/summit/pytorch-1.0-p3/
source source_to_run_pytorch1.0-p3


cd     /gpfs/alpine/med106/proj-shared/aclyde/MolecularAttention
export TORCH_HOME=/gpfs/alpine/med106/proj-shared/aclyde/torch_cache/
jsrun -n132 -g1 -a1 -c7 python qm8_summit_tune.py
