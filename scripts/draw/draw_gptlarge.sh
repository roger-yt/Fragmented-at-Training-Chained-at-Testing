export CUDA_VISIBLE_DEVICES="0"



parent_dir="data_and_models/gptlarge"

python draw.py \
     --config-name config_gptlarge.yaml\
    draw.parent_dir=$parent_dir\
    draw.mode=main

python draw.py \
     --config-name config_gptlarge.yaml\
    draw.parent_dir=$parent_dir\
    draw.mode=ratio
