#################################################################
# Edit the following paths to match your setup
BASE_DIR='/path/to/parent/folder'
NUSCENES_PATH='/datasets/nuscenes'
# Model related stuff
MODEL_NAME='UniAD' # UniAD example
MODEL_FOLDER=$BASE_DIR/$MODEL_NAME # UniAD example
MODEL_CHECKPOINT_PATH='checkpoints/uniad_base_e2e.pth' # UniAD example
MODEL_CFG_PATH='projects/configs/stage2_e2e/inference_e2e.py' # UniAD example
MODEL_IMAGE='uniad:latest' # UniAD example
# Rendering related stuff
RENDERING_FOLDER=$BASE_DIR/'neurad-studio'
RENDERING_CHECKPOITNS_PATH='checkpoints'
RENDERING_IMAGE='neurad:latest'
# NCAP related stuff
NCAP_FOLDER=$BASE_DIR/'neuro-ncap'
NCAP_IMAGE='ncap:latest'

# Evaluation default values, set to lower for debugging
RUNS=50

#################################################################

# SLURM related stuff
TIME_NOW=$(date +"%Y-%m-%d_%H-%M-%S")


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


for SCENARIO in "stationary" "frontal" "side"; do
    array_file=ncap_slurm_array_$SCENARIO
    id_to_seq=scripts/arrays/${array_file}.txt

    if [ $SCENARIO == "stationary" ]; then
        num_scenarios=10
    elif [ $SCENARIO == "frontal" ]; then
        num_scenarios=5
    elif [ $SCENARIO == "side" ]; then
        num_scenarios=5
    fi
    for i in $(seq 1 $num_scenarios); do
        sequence=$(awk -v ArrayTaskID=$i '$1==ArrayTaskID {print $2}' $id_to_seq)
        if [ -z $sequence ]; then
            echo "undefined sequence"
            exit 0
        fi
        echo "Running scenario $SCENARIO with sequence $sequence"
        BASE_DIR=$BASE_DIR\
         NUSCENES_PATH=$NUSCENES_PATH\
         MODEL_NAME=$MODEL_NAME\
         MODEL_FOLDER=$MODEL_FOLDER\
         MODEL_CHECKPOINT_PATH=$MODEL_CHECKPOINT_PATH\
         MODEL_CFG_PATH=$MODEL_CFG_PATH\
         MODEL_IMAGE=$MODEL_IMAGE\
         RENDERING_FOLDER=$RENDERING_FOLDER\
         RENDERING_CHECKPOITNS_PATH=$RENDERING_CHECKPOITNS_PATH\
         RENDERING_IMAGE=$RENDERING_IMAGE\
         NCAP_FOLDER=$NCAP_FOLDER\
         NCAP_IMAGE=$NCAP_IMAGE\
         TIME_NOW=$TIME_NOW\
         scripts/_docker_compose_release.sh $sequence $SCENARIO --scenario-category=$SCENARIO --runs $RUNS
        exit 0
    done
done
