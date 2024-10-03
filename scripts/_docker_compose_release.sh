#!/bin/bash
seq=${1:?"No sequence specified"}
output_name=${2:?"No output name given (for logging)"}

# loop over all remaining args and check if "--spoof-renderer or --spoof_renderer" is in  ${@:4}
# if it is, set RENDERER_ARGS="--spoof-renderer" and remove it from the list of args
SHOULD_START_RENDERER=true
for arg in ${@:3}; do
  if [[ $arg == "--spoof-renderer" || $arg == "--spoof_renderer" ]]; then
    SHOULD_START_RENDERER=false
  fi
done

# loop over all remaining args and check if "--spoof-renderer or --spoof_renderer" is in  ${@:4}
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


# TODO: how do we set the nuscenes path correctly here?
if [ $SHOULD_START_RENDERER == true ]; then
  echo "Running NeuRAD service in background..."
  docker run --name renderer --rm --gpus all \
    -v $RENDERING_FOLDER:/neurad_studio \
    -v $NUSCENES_PATH:/neurad_studio/data/nuscenes \
    --network host \
    -e PYTHONPATH=. \
    -w /neurad_studio \
    $RENDERING_IMAGE \
    python -u nerfstudio/scripts/closed_loop/main.py \
    --port $renderer_port \
    --load-config $RENDERING_CHECKPOITNS_PATH/$seq/config.yml \
    --adjust_pose \
    $RENDERER_ARGS \
    &
fi

if [ $SHOULD_START_MODEL == true ]; then
  echo "Running $MODEL_NAME service in background..."
  docker run --name model --rm --gpus all \
    -v $MODEL_FOLDER:/model \
    -w /model \
    --network host \
    -e PYTHONPATH=. \
    $MODEL_IMAGE \
    python -u inference/server.py \
    --port $model_port \
    --config_path $MODEL_CFG_PATH \
    --checkpoint_path $MODEL_CHECKPOINT_PATH \
    $MODEL_ARGS \
    &
fi

echo "Running neuro-ncap in foreground..."
docker run --rm --gpus all \
  -v $PWD:/neuro_ncap \
  -v $NUSCENES_PATH:$NUSCENES_PATH \
  -w /neuro_ncap \
  --network host \
  $NCAP_IMAGE \
  python -u main.py \
  --engine.renderer.port $renderer_port \
  --engine.model.port $model_port \
  --engine.dataset.data_root $NUSCENES_PATH \
  --engine.dataset.version v1.0-trainval \
  --engine.dataset.sequence $seq \
  --engine.logger.log-dir outoutput/$TIME_NOW/$output_name-$seq \
  ${@:3}

# Kill the background processes
echo "Killing background processes..."
if [ $SHOULD_START_RENDERER == true ]; then
 docker kill renderer
fi
if [ $SHOULD_START_MODEL == true ]; then
 docker kill model
fi
