#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 #uncomment to specify GPUs
enviroment="NoWag"

models=("meta-llama/Llama-2-7b-hf"
    # "meta-llama/Llama-2-13b-hf"
    # "meta-llama/Llama-2-70b-hf"
    # "meta-llama/Meta-Llama-3-8B"
    # "meta-llama/Meta-Llama-3-70B"
)

for model in "${models[@]}"; do
    echo "===========Quantizing $model =========="
    cmd="python -u NoWag.py run_name=2bit_vq compress=vq resume=True"
    echo "running command: $cmd"
    conda run -n $enviroment --live-stream $cmd
done