export CUDA_VISIBLE_DEVICES="0"



parent_dir="data_and_models/gptsmall"

python draw_hydra.py \
     --config-name config_gptsmall.yaml\
    draw.parent_dir=$parent_dir\
    draw.mode=main

python draw_hydra.py \
     --config-name config_gptsmall.yaml\
    draw.parent_dir=$parent_dir\
    draw.mode=ratio
