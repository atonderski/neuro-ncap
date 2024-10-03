#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --time 02:00:00
#SBATCH --array=1-10
#
# This script is a wrapper around the _slurm_compose_release.sh script
scenario=${1:?"No scenario given"}
scripts/_slurm_compose_release.sh ncap_slurm_array_$scenario $scenario --scenario-category=$scenario ${@:2}
#
#EOF
