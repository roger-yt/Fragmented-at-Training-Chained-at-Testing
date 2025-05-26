export CUDA_VISIBLE_DEVICES="0"



parent_dir="data_and_models/gptlarge"

python draw_hydra.py \
     --config-name config_gptlarge.yaml\
    draw.parent_dir=$parent_dir\
    draw.mode=main

python draw_hydra.py \
     --config-name config_gptlarge.yaml\
    draw.parent_dir=$parent_dir\
    draw.mode=ratio
