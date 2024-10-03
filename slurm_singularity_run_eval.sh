
#################################################################
# Edit the following paths to match your setup
export BASE_DIR='/proj/agp/users/x_adato'
export NUSCENES_PATH='/proj/adas-data/data/nuscenes'
# Model related stuff
export MODEL_NAME='UniAD' # UniAD example
export MODEL_FOLDER=$BASE_DIR/$MODEL_NAME # UniAD example
export MODEL_CHECKPOINT_PATH=$MODEL_FOLDER/'checkpoints/uniad_base_e2e.pth' # UniAD example
export MODEL_CFG_PATH=$MODEL_FOLDER/'projects/configs/stage2_e2e/inference_e2e.py' # UniAD example
export MODEL_CONTAINER=$MODEL_FOLDER/'uniad.sif' # UniAD example
# Rendering related stuff
export RENDERING_FOLDER=$BASE_DIR/'neurad-studio'
export RENDERING_CHECKPOITNS_PATH=$RENDERING_FOLDER/'checkpoints'
export RENDERING_CONTAINER=$RENDERING_FOLDER/'neurad-studio.sif'
# NCAP related stuff
export NCAP_FOLDER=$BASE_DIR/'neuro-ncap'
export NCAP_CONTAINER=$NCAP_FOLDER/'neuro-ncap.sif'

# Evaluation default values, set to lower for debugging
export RUNS=50

#################################################################

# SLURM related stuff
export TIME_NOW=$(date +"%Y-%m-%d_%H-%M-%S")
export SLURM_OUTPUT_FOLDER=$BASE_DIR/slurm_logs/$TIME_NOW/%A_%a.out

# if folder does not exist, create it
if [ ! -d $BASE_DIR/slurm_logs/$TIME_NOW ]; then
    mkdir -p $BASE_DIR/slurm_logs/$TIME_NOW
fi


# assert we are standing in the right folder, which is NCAP folder
if [ $PWD != $NCAP_FOLDER ]; then
    echo "Please run this script from the NCAP folder"
    exit 1
fi

# assert all the other folders are present
if [ ! -d $MODEL_FOLDER ]; then
    echo "Model folder not found"
    exit 1
fi
if [ ! -d $RENDERING_FOLDER ]; then
    echo "Rendering folder not found"
    exit 1
fi

# assert all singularity files exist
if [ ! -f $MODEL_CONTAINER ]; then
    echo "Model container file not found"
    exit 1
fi
if [ ! -f $RENDERING_CONTAINER ]; then
    echo "Rendering container file not found"
    exit 1
fi
if [ ! -f $NCAP_CONTAINER ]; then
    echo "NCAP container file not found"
    exit 1
fi

for SCENARIO in "stationary" "frontal" "side"; do
  sbatch $SLURM_ARGS -o $SLURM_OUTPUT_FOLDER scripts/slurm_compose_scenario_release.sh $SCENARIO --runs $RUNS
done
