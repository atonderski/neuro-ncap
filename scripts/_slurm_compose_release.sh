#!/bin/bash
array_file=${1:?"No array file given"}
output_name=${2:?"No output name given (for logging)"}


# loop over all remaining args and check if "--spoof-renderer or --spoof_renderer" is in  ${@:4}
# if it is, set RENDERER_ARGS="--spoof-renderer" and remove it from the list of args
SHOULD_START_RENDERER=true
for arg in ${@:3}; do
  if [[ $arg == "--spoof-renderer" || $arg == "--spoof_renderer" ]]; then
    SHOULD_START_RENDERER=false
  fi
done

# loop over all remaining args and check if "--spoof-renderer or --spoof_renderer" is in  ${@:3}
# if it is, set RENDERER_ARGS="--spoof-renderer" and remove it from the list of args
SHOULD_START_MODEL=true
for arg in ${@:3}; do
  if [[ $arg == "--spoof-model" || $arg == "--spoof_model" ]]; then
    SHOULD_START_MODEL=false
  fi
done

# find two free ports
find_free_port() {
  python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}

renderer_port=$(find_free_port)
model_port=$(find_free_port)

# figure out the array sequence (and abort if it is empty)
id_to_seq=scripts/arrays/${array_file}.txt
seq=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $id_to_seq)
[[ -z $seq ]] && echo "undefined sequence" && exit 0

# TODO: how do we set the nuscenes path correctly here?
if [ $SHOULD_START_RENDERER == true ]; then
  echo "Running NeuRAD service in background..."
  singularity exec --nv \
    --bind $RENDERING_FOLDER:/neurad_studio \
    --bind $NUSCENES_PATH:/neurad_studio/data/nuscenes \
    --pwd /neurad_studio \
    --env PYTHONPATH=. \
    $RENDERING_CONTAINER \
    python -u nerfstudio/scripts/closed_loop/main.py \
    --port $renderer_port \
    --load-config $RENDERING_CHECKPOITNS_PATH/$seq/config.yml \
    --adjust_pose \
    $RENDERER_ARGS \
    &
fi

if [ $SHOULD_START_MODEL == true ]; then
  echo "Running $MODEL_NAME service in background..."
  singularity exec --nv \
    --bind $MODEL_FOLDER:/$MODEL_NAME \
    --pwd /$MODEL_NAME \
    --env PYTHONPATH=. \
    $MODEL_CONTAINER \
    python -u inference/server.py \
    --port $model_port \
    --config_path $MODEL_CFG_PATH \
    --checkpoint_path $MODEL_CHECKPOINT_PATH \
    $MODEL_ARGS \
    &
fi

echo "Running neuro-ncap in foreground..."
singularity exec --nv \
  --bind $NCAP_FOLDER:/neuro-ncap \
  --bind $NUSCENES_PATH:$NUSCENES_PATH \
  --pwd /neuro-ncap \
  $NCAP_CONTAINER \
  python -u main.py \
  --engine.renderer.port $renderer_port \
  --engine.model.port $model_port \
  --engine.dataset.data_root $NUSCENES_PATH \
  --engine.dataset.version v1.0-trainval \
  --engine.dataset.sequence $seq \
  --engine.logger.log-dir outputs/$TIME_NOW/$output_name-$seq \
  ${@:3}


#
#EOF
